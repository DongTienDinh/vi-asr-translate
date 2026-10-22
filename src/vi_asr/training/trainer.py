"""Main DDP Trainer for Vietnamese ASR on Kaggle Dual T4 (Section 17)."""

from __future__ import annotations

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
from vi_asr.training.checkpoint import capture_rng_states, save_training_checkpoint
from vi_asr.training.distributed import (
    all_reduce_sample_count,
    get_rank,
    get_world_size,
    is_main_process,
    rescale_gradients_by_sample_count,
)
from vi_asr.training.dataloader import reseed_loader_for_epoch
from vi_asr.training.losses import MultiTaskASRLoss
from vi_asr.utils.logging import setup_logger

logger = setup_logger(__name__)


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
        runtime_cfg = self.config.get("runtime", {})
        self.base_seed = int(self.config.get("experiment", {}).get("seed", 42))
        self.progress_log_interval = max(1, int(runtime_cfg.get("progress_log_interval_updates", 20)))
        self.non_blocking_transfers = bool(runtime_cfg.get("non_blocking_transfers", True))

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        scheduler: Optional[Any] = None,
        max_updates: Optional[int] = None,
    ) -> float:
        self.model.train()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        microstep_count = 0
        optimizer_updates = 0
        epoch_started = time.perf_counter()
        progress_started = epoch_started
        last_batch_finished = epoch_started
        data_wait_seconds = 0.0
        progress_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        progress_microsteps = 0
        progress_global_samples = 0

        # Accumulated samples across microsteps within one accumulation window
        accum_samples_local = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            data_wait_seconds += time.perf_counter() - last_batch_finished
            # Move tensors to device
            for k in ("feats", "feat_lengths", "ctc_targets", "target_lengths", "teacher_states", "teacher_lengths"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device, non_blocking=self.non_blocking_transfers)

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

            detached_loss = raw_loss.detach().to(dtype=torch.float64)
            total_loss += detached_loss
            progress_loss += detached_loss
            microstep_count += 1
            progress_microsteps += 1

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
                progress_global_samples += W

                should_log = (
                    optimizer_updates % self.progress_log_interval == 0
                    or batch_idx == len(train_loader) - 1
                )
                if should_log:
                    now = time.perf_counter()
                if self.rank == 0 and should_log:
                    window_seconds = max(now - progress_started, 1e-9)
                    epoch_seconds = max(now - epoch_started, 1e-9)
                    mean_loss = float(progress_loss.item()) / max(1, progress_microsteps)
                    lr = float(self.optimizer.param_groups[0]["lr"])
                    memory_gib = (
                        torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    logger.info(
                        "Epoch %d progress %d/%d microsteps (%.1f%%) | update=%d | "
                        "loss=%.4f | %.2f global samples/s | data_wait=%.1f%% | "
                        "lr=%.3e | peak_gpu=%.2f GiB",
                        epoch,
                        batch_idx + 1,
                        len(train_loader),
                        100.0 * (batch_idx + 1) / max(1, len(train_loader)),
                        self.global_updates,
                        mean_loss,
                        progress_global_samples / window_seconds,
                        100.0 * data_wait_seconds / epoch_seconds,
                        lr,
                        memory_gib,
                    )
                if should_log:
                    progress_started = now
                    progress_loss.zero_()
                    progress_microsteps = 0
                    progress_global_samples = 0

                if max_updates is not None and self.global_updates >= max_updates:
                    break

            last_batch_finished = time.perf_counter()

        avg_loss = float(total_loss.item()) / max(1, microstep_count)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Evaluate validation set, compute loss and optional WER/CER."""
        raw_m = self.model.module if hasattr(self.model, "module") else self.model
        self.model.eval()
        total_val_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        steps = 0
        all_hyps: List[str] = []
        all_refs: List[str] = []

        for batch in val_loader:
            for k in ("feats", "feat_lengths", "ctc_targets", "target_lengths", "teacher_states", "teacher_lengths"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device, non_blocking=self.non_blocking_transfers)

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
            total_val_loss += loss_dict["loss"].detach().to(dtype=torch.float64)
            steps += 1

            if tokenizer is not None:
                asr_net = raw_m.asr_model if hasattr(raw_m, "asr_model") else raw_m
                blank_id = int(getattr(asr_net, "blank_id", 0))
                preds = torch.argmax(out["logits"], dim=-1)
                hyp_ids = []
                for i in range(preds.size(0)):
                    valid_t = int(out["output_lengths"][i].item())
                    path = preds[i, :valid_t].tolist()
                    collapsed = []
                    previous = -1
                    for token_id in path:
                        if token_id != previous:
                            if token_id != blank_id:
                                collapsed.append(token_id)
                            previous = token_id
                    hyp_ids.append(collapsed)

                refs = batch.get("transcripts") or []
                for h_ids, ref_str in zip(hyp_ids, refs):
                    all_hyps.append(tokenizer.decode(h_ids) if hasattr(tokenizer, "decode") else "")
                    all_refs.append(str(ref_str))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_stats = torch.stack(
                [total_val_loss, torch.tensor(float(steps), dtype=torch.float64, device=self.device)]
            )
            torch.distributed.all_reduce(loss_stats, op=torch.distributed.ReduceOp.SUM)
            total_val_loss = loss_stats[0]
            steps = int(loss_stats[1].item())

            gathered: List[Any] = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(gathered, (all_hyps, all_refs))
            all_hyps = [hyp for rank_hyps, _ in gathered for hyp in rank_hyps]
            all_refs = [ref for _, rank_refs in gathered for ref in rank_refs]

        avg_loss = float(total_val_loss.item()) / max(1, steps)
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
        start_epoch: int = 1,
        epochs_this_session: Optional[int] = None,
        resume_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run complete training loop across epochs or until max_updates."""
        resume_metrics = resume_metrics or {}
        best_wer = float(resume_metrics.get("best_wer", float("inf")))
        best_cer = float(resume_metrics.get("best_cer", float("inf")))
        best_loss = float(resume_metrics.get("best_loss", float("inf")))
        best_epoch = int(resume_metrics.get("best_epoch", 0))

        final_epoch = self.epochs
        if epochs_this_session is not None:
            if epochs_this_session <= 0:
                raise ValueError("epochs_this_session must be positive")
            final_epoch = min(self.epochs, start_epoch + epochs_this_session - 1)

        for epoch in range(start_epoch, final_epoch + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            worker_seed = reseed_loader_for_epoch(
                train_loader,
                base_seed=self.base_seed,
                epoch=epoch,
                rank=self.rank,
            )
            if self.rank == 0:
                logger.info(
                    "Epoch %d/%d starting: %d microsteps, worker_seed=%d, progress every %d updates",
                    epoch,
                    self.epochs,
                    len(train_loader),
                    worker_seed,
                    self.progress_log_interval,
                )
            train_loss = self.train_epoch(epoch, train_loader, scheduler, max_updates=max_updates)
            if self.rank == 0:
                logger.info("Epoch %d/%d completed. Avg Train Loss: %.4f", epoch, self.epochs, train_loss)

            val_metrics: Dict[str, Any] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, tokenizer=tokenizer)
                if self.rank == 0:
                    logger.info(
                        "Epoch %d Validation: Loss: %.4f | WER: %.4f | CER: %.4f",
                        epoch,
                        val_metrics["val_loss"],
                        val_metrics.get("wer", 1.0),
                        val_metrics.get("cer", 1.0),
                    )

            current_metric = float(val_metrics.get("wer", train_loss))
            current_cer = float(val_metrics.get("cer", float("inf")))
            is_best = current_metric < best_wer or (
                current_metric == best_wer and current_cer < best_cer
            )
            if is_best:
                best_wer = current_metric
                best_cer = current_cer
                best_loss = val_metrics.get("val_loss", train_loss)
                best_epoch = epoch

            local_rng = capture_rng_states()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rng_states_by_rank: List[Any] = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(rng_states_by_rank, local_rng)
            else:
                rng_states_by_rank = [local_rng]

            if is_main_process():
                save_training_checkpoint(
                    checkpoint_dir=self.output_dir,
                    epoch=epoch,
                    global_step=self.global_updates,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=scheduler,
                    metrics={
                        "train_loss": train_loss,
                        **val_metrics,
                        "best_wer": best_wer,
                        "best_cer": best_cer,
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                    },
                    config=self.config,
                    is_best=is_best,
                    rng_states_by_rank=rng_states_by_rank,
                )

            if max_updates is not None and self.global_updates >= max_updates:
                logger.info("Reached max_updates (%d). Stopping training.", max_updates)
                break

        if final_epoch < self.epochs:
            logger.info(
                "Session epoch limit reached at %d/%d. Save output and resume from checkpoint_latest.pt.",
                final_epoch,
                self.epochs,
            )

        return {
            "best_wer": best_wer,
            "best_cer": best_cer,
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
