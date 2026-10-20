"""Main DDP Trainer for Vietnamese ASR on Kaggle Dual T4 (Section 17)."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from vi_asr.data.collate import asr_collate_fn
from vi_asr.training.distributed_sampler import GlobalDurationSampler, ValidationShardSampler
from vi_asr.models.alignment import TokenFramePoolingAlignment
from vi_asr.models.ctc_asr import ViPViCTCASR
from vi_asr.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from vi_asr.training.distributed import (
    all_reduce_sample_count,
    get_rank,
    get_world_size,
    is_main_process,
    rescale_gradients_by_sample_count,
)
from vi_asr.training.losses import MultiTaskASRLoss

logger = logging.getLogger(__name__)


def create_warmup_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup followed by linear decay scheduler."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


class DDPASTrainer:
    """Trainer managing dual-GPU DDP training, AMP FP16, and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "runs/main",
        accumulation_steps: int = 8,
        nominal_effective_batch: int = 32,
        max_grad_norm: float = 5.0,
        device: Optional[torch.device] = None,
    ):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rank = get_rank()
        self.world_size = get_world_size()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.accumulation_steps = accumulation_steps
        self.nominal_effective_batch = nominal_effective_batch
        self.max_grad_norm = max_grad_norm

        # Precision & Scaler
        use_cuda = self.device.type == "cuda"
        self.use_amp = use_cuda
        if use_cuda:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Extract training hyperparams from config
        train_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})

        lambda_token = float(loss_cfg.get("lambda_token", 0.0))
        lambda_region = float(loss_cfg.get("lambda_region", 0.0))
        self.loss_fn = MultiTaskASRLoss(
            lambda_token=lambda_token,
            lambda_region=lambda_region,
            ctc_blank_id=0,
        ).to(self.device)

        # Separate LR for encoder vs heads
        enc_lr = float(train_cfg.get("encoder_lr", 1e-5))
        heads_lr = float(train_cfg.get("heads_lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))

        params = []
        raw_m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_m, "asr_model") and hasattr(raw_m.asr_model, "encoder_adapter"):
            enc_params = list(raw_m.asr_model.encoder_adapter.parameters())
            head_params = [p for p in raw_m.parameters() if not any(p is ep for ep in enc_params)]
            params = [
                {"params": enc_params, "lr": enc_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": heads_lr, "weight_decay": weight_decay},
            ]
        else:
            params = [{"params": raw_m.parameters(), "lr": heads_lr, "weight_decay": weight_decay}]

        self.optimizer = AdamW(params)
        self.epochs = int(train_cfg.get("epochs", 20))
        self.global_updates = 0

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        scheduler: Optional[Any] = None,
        max_updates: Optional[int] = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        microstep_count = 0
        optimizer_updates = 0

        # Accumulated samples across microsteps within one accumulation window
        accum_samples_local = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            for k in ("feats", "feat_lengths", "ctc_targets", "target_lengths", "teacher_states", "teacher_lengths"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            local_b = batch["feats"].size(0)
            accum_samples_local += local_b

            is_accum_boundary = (microstep_count % self.accumulation_steps == self.accumulation_steps - 1) or (
                batch_idx == len(train_loader) - 1
            )

            # Use no_sync on DDP during intermediate accumulation steps
            sync_context = self.model.no_sync() if (hasattr(self.model, "no_sync") and not is_accum_boundary) else nullcontext()

            with sync_context:
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = self.model(batch)
                        loss_dict = self.loss_fn(
                            ctc_logits=out["logits"],
                            ctc_output_lengths=out["output_lengths"],
                            ctc_targets=batch["ctc_targets"],
                            ctc_target_lengths=batch["target_lengths"],
                            predicted_teacher=out.get("predicted_teacher"),
                            target_teacher=batch.get("teacher_states"),
                            token_lengths=batch.get("teacher_lengths"),
                        )
                        raw_loss = loss_dict["loss"]
                        scaled_loss = raw_loss / self.accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                else:
                    out = self.model(batch)
                    loss_dict = self.loss_fn(
                        ctc_logits=out["logits"],
                        ctc_output_lengths=out["output_lengths"],
                        ctc_targets=batch["ctc_targets"],
                        ctc_target_lengths=batch["target_lengths"],
                        predicted_teacher=out.get("predicted_teacher"),
                        target_teacher=batch.get("teacher_states"),
                        token_lengths=batch.get("teacher_lengths"),
                    )
                    raw_loss = loss_dict["loss"]
                    scaled_loss = raw_loss / self.accumulation_steps
                    scaled_loss.backward()

            total_loss += raw_loss.item()
            microstep_count += 1

            if is_accum_boundary:
                # 1. All-reduce total samples W across all ranks in this update
                W = all_reduce_sample_count(accum_samples_local, self.device)

                # 2. Rescale gradients by (R * D / W)
                raw_m = self.model.module if hasattr(self.model, "module") else self.model
                rescale_gradients_by_sample_count(
                    raw_m.parameters(),
                    effective_sample_count=W,
                    nominal_sample_count=self.nominal_effective_batch,
                )

                # 3. Unscale, clip grad norm, step optimizer
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_m.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(raw_m.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                self.optimizer.zero_grad()
                accum_samples_local = 0
                optimizer_updates += 1
                self.global_updates += 1

                if max_updates is not None and self.global_updates >= max_updates:
                    break

        avg_loss = total_loss / max(1, microstep_count)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Evaluate validation set, compute loss and optional WER/CER."""
        raw_m = self.model.module if hasattr(self.model, "module") else self.model
        self.model.eval()
        total_val_loss = 0.0
        steps = 0
        all_hyps: List[str] = []
        all_refs: List[str] = []

        for batch in val_loader:
            for k in ("feats", "feat_lengths", "ctc_targets", "target_lengths", "teacher_states", "teacher_lengths"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            out = self.model(batch)
            loss_dict = self.loss_fn(
                ctc_logits=out["logits"],
                ctc_output_lengths=out["output_lengths"],
                ctc_targets=batch["ctc_targets"],
                ctc_target_lengths=batch["target_lengths"],
                predicted_teacher=out.get("predicted_teacher"),
                target_teacher=batch.get("teacher_states"),
                token_lengths=batch.get("teacher_lengths"),
            )
            total_val_loss += loss_dict["loss"].item()
            steps += 1

            if tokenizer is not None:
                asr_net = raw_m.asr_model if hasattr(raw_m, "asr_model") else raw_m
                if hasattr(asr_net, "transcribe"):
                    hyp_ids = asr_net.transcribe(batch["feats"], batch["feat_lengths"])
                else:
                    logits = out["logits"]
                    preds = torch.argmax(logits, dim=-1)
                    hyp_ids = [preds[i, : int(out["output_lengths"][i])].tolist() for i in range(preds.size(0))]

                refs = batch.get("transcripts") or []
                for h_ids, ref_str in zip(hyp_ids, refs):
                    all_hyps.append(tokenizer.decode(h_ids) if hasattr(tokenizer, "decode") else "")
                    all_refs.append(str(ref_str))

        avg_loss = total_val_loss / max(1, steps)
        metrics: Dict[str, Any] = {"val_loss": avg_loss}

        if all_hyps and all_refs:
            from vi_asr.evaluation.metrics import compute_wer_cer
            wer_cer = compute_wer_cer(all_hyps, all_refs)
            metrics["wer"] = wer_cer["wer"]
            metrics["cer"] = wer_cer["cer"]
        else:
            metrics["wer"] = 1.0
            metrics["cer"] = 1.0

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        max_updates: Optional[int] = None,
        train_sampler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run complete training loop across epochs or until max_updates."""
        best_wer = float("inf")
        best_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            logger.info("Epoch %d/%d starting...", epoch, self.epochs)
            train_loss = self.train_epoch(epoch, train_loader, scheduler, max_updates=max_updates)
            logger.info("Epoch %d/%d completed. Avg Train Loss: %.4f", epoch, self.epochs, train_loss)

            val_metrics: Dict[str, Any] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, tokenizer=tokenizer)
                logger.info(
                    "Epoch %d Validation: Loss: %.4f | WER: %.4f | CER: %.4f",
                    epoch,
                    val_metrics["val_loss"],
                    val_metrics.get("wer", 1.0),
                    val_metrics.get("cer", 1.0),
                )

            current_metric = val_metrics.get("wer", train_loss)
            is_best = current_metric < best_wer
            if is_best:
                best_wer = current_metric
                best_loss = val_metrics.get("val_loss", train_loss)
                best_epoch = epoch

            if is_main_process():
                save_training_checkpoint(
                    checkpoint_dir=self.output_dir,
                    epoch=epoch,
                    global_step=self.global_updates,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=scheduler,
                    metrics={"train_loss": train_loss, **val_metrics},
                    config=self.config,
                    is_best=is_best,
                )

            if max_updates is not None and self.global_updates >= max_updates:
                logger.info("Reached max_updates (%d). Stopping training.", max_updates)
                break

        return {
            "best_wer": best_wer,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "checkpoint_best": str(self.output_dir / "checkpoint_best.pt"),
            "checkpoint_latest": str(self.output_dir / "checkpoint_latest.pt"),
        }


class nullcontext:
    """Fallback context manager when no_sync is not needed."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
