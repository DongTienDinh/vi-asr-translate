"""Unit tests for ViP-VL ChunkFormer encoder adapter, Pure CTC ASR model, and checkpoint loader (Gate G2)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from vi_asr.models.checkpoint import load_vip_vl_pretrained
from vi_asr.models.ctc_asr import CTCHead, ViPViCTCASR
from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter


class TestViPVlEncoderAndCTC(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _create_small_adapter(self, num_blocks: int = 2) -> ChunkFormerEncoderAdapter:
        """Create lightweight ChunkFormer encoder adapter for fast CPU testing."""
        return ChunkFormerEncoderAdapter(
            input_size=80,
            output_size=512,
            attention_heads=8,
            linear_units=1024,
            num_blocks=num_blocks,
            dropout_rate=0.0,
            encoder_layerdrop=0.0,
            cnn_module_kernel=15,
            dynamic_conv=True,
            gradient_checkpointing=False,
        )

    def test_encoder_adapter_shapes_and_subsampling(self):
        """Verify encoder output shape [B, T, 512], subsampling lengths, and valid mask semantics."""
        adapter = self._create_small_adapter(num_blocks=2)
        adapter.eval()

        # Batch of 3 samples with lengths 100, 75, 50 frames
        feats = torch.randn(3, 100, 80)
        lengths = torch.tensor([100, 75, 50], dtype=torch.int64)

        H, output_lengths, valid_mask = adapter.encode(feats, lengths)

        # Output shape: [3, 11, 512] for 100 input frames after 8x subsampling
        self.assertEqual(H.shape[0], 3)
        self.assertEqual(H.shape[2], 512)
        T_sub = H.shape[1]

        # Check output lengths and mask
        self.assertEqual(output_lengths.shape, (3,))
        self.assertEqual(output_lengths.dtype, torch.int64)
        self.assertEqual(valid_mask.shape, (3, T_sub))
        self.assertEqual(valid_mask.dtype, torch.bool)

        # Ensure lengths match active mask count
        computed_lengths = valid_mask.sum(dim=-1).to(torch.int64)
        self.assertTrue(torch.equal(output_lengths, computed_lengths))

        # Sample 0 (100 frames) should be fully valid
        self.assertTrue(torch.all(valid_mask[0]))

        # Sample 2 (50 frames) should have trailing padding (False)
        self.assertFalse(torch.all(valid_mask[2]))
        valid_count_2 = int(output_lengths[2].item())
        self.assertTrue(torch.all(valid_mask[2, :valid_count_2]))
        self.assertTrue(torch.all(~valid_mask[2, valid_count_2:]))

    def test_ctc_head_shapes_and_init(self):
        """Verify CTCHead projection shape and zero-bias initialization."""
        head = CTCHead(hidden_size=512, vocab_size=1024, dropout=0.0)
        self.assertEqual(head.linear.weight.shape, (1024, 512))
        self.assertEqual(head.linear.bias.shape, (1024,))

        # Verify zero-bias initialization
        self.assertTrue(torch.all(head.linear.bias == 0.0))

        # Forward pass
        H = torch.randn(2, 10, 512)
        logits = head(H)
        self.assertEqual(logits.shape, (2, 10, 1024))

    def test_ctc_asr_forward_and_transcribe(self):
        """Verify ViPViCTCASR forward pass and greedy transcribe with CTC collapse."""
        adapter = self._create_small_adapter(num_blocks=2)
        model = ViPViCTCASR(adapter, vocab_size=120, blank_id=0)

        feats = torch.randn(2, 80, 80)
        lengths = torch.tensor([80, 60], dtype=torch.int64)

        logits, output_lengths, valid_mask = model(feats, lengths)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[2], 120)
        self.assertEqual(output_lengths.shape, (2,))

        # Test greedy transcription
        hypotheses = model.transcribe(feats, lengths)
        self.assertEqual(len(hypotheses), 2)
        self.assertIsInstance(hypotheses[0], list)
        self.assertIsInstance(hypotheses[1], list)

        # Verify no blank token (0) in hypotheses
        self.assertNotIn(0, hypotheses[0])
        self.assertNotIn(0, hypotheses[1])

    def test_per_example_ctc_loss_contract(self):
        """Verify CTC loss is computed in FP32 and normalized per example by target length U_i."""
        adapter = self._create_small_adapter(num_blocks=2)
        model = ViPViCTCASR(adapter, vocab_size=50, blank_id=0)

        batch_size = 2
        T_out = 15
        logits = torch.randn(batch_size, T_out, 50, requires_grad=True)
        output_lengths = torch.tensor([15, 12], dtype=torch.int64)

        # 2 samples with targets: sample 1 has 3 tokens, sample 2 has 2 tokens
        targets = torch.tensor([[5, 6, 7], [8, 9, 0]], dtype=torch.int64)
        target_lengths = torch.tensor([3, 2], dtype=torch.int64)

        mean_loss, per_example_loss = model.compute_ctc_loss(logits, output_lengths, targets, target_lengths)

        self.assertEqual(per_example_loss.shape, (2,))
        self.assertEqual(mean_loss.dtype, torch.float32)
        self.assertEqual(per_example_loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(mean_loss))
        self.assertAlmostEqual(mean_loss.item(), per_example_loss.mean().item(), places=5)

        # Verify backward gradient flow to logits
        mean_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_checkpoint_loader_strict_success_and_report(self):
        """Verify load_vip_vl_pretrained strictly matches all backbone keys and generates load report (Gate G2)."""
        adapter = self._create_small_adapter(num_blocks=2)
        model = ViPViCTCASR(adapter, vocab_size=100, blank_id=0)

        # Create synthetic checkpoint simulating khanhld/vip-vl-base-vie:
        # contains 'encoder.<key>' for all encoder parameters, plus SSL heads ('quantizer.*', 'final_proj.*')
        synth_state_dict = {}
        for k, v in adapter.encoder.state_dict().items():
            synth_state_dict[f"encoder.{k}"] = v.clone()

        # Add simulated upstream pretraining keys that should be safely ignored
        synth_state_dict["quantizer.codebook"] = torch.randn(1024, 16)
        synth_state_dict["final_proj.weight"] = torch.randn(512, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "pytorch_model.pt"
            torch.save({"model": synth_state_dict}, ckpt_path)

            report_path = Path(tmpdir) / "load_report.json"
            report = load_vip_vl_pretrained(
                model_or_encoder=model,
                checkpoint_path_or_repo=ckpt_path,
                report_path=report_path,
            )

            # Gate G2 assertions
            self.assertTrue(report["strict_backbone_passed"])
            self.assertEqual(len(report["missing_keys"]), 0)
            self.assertEqual(len(report["unexpected_keys"]), 0)
            self.assertEqual(report["ignored_upstream_keys_count"], 2)
            self.assertGreater(report["total_encoder_parameters"], 0)
            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                saved_report = json.load(f)
            self.assertEqual(saved_report["gate"], "G2")
            self.assertTrue(saved_report["strict_backbone_passed"])

    def test_checkpoint_loader_missing_key_fails(self):
        """Verify that any missing encoder key raises RuntimeError and does not pass silently."""
        adapter = self._create_small_adapter(num_blocks=2)

        # Synthetic state dict missing one key
        synth_state_dict = {}
        target_keys = list(adapter.encoder.state_dict().keys())
        for k in target_keys[:-1]:  # omit the last key
            synth_state_dict[f"encoder.{k}"] = adapter.encoder.state_dict()[k].clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "pytorch_model.pt"
            torch.save(synth_state_dict, ckpt_path)

            with self.assertRaises(RuntimeError) as ctx:
                load_vip_vl_pretrained(
                    model_or_encoder=adapter,
                    checkpoint_path_or_repo=ckpt_path,
                )
            self.assertIn("Missing", str(ctx.exception))

    def test_checkpoint_loader_shape_mismatch_fails(self):
        """Verify that any parameter shape mismatch raises RuntimeError immediately."""
        adapter = self._create_small_adapter(num_blocks=2)

        synth_state_dict = {}
        for k, v in adapter.encoder.state_dict().items():
            synth_state_dict[f"encoder.{k}"] = v.clone()

        # Corrupt the shape of the first parameter
        first_key = f"encoder.{list(adapter.encoder.state_dict().keys())[0]}"
        synth_state_dict[first_key] = torch.randn(999, 999)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "pytorch_model.pt"
            torch.save(synth_state_dict, ckpt_path)

            with self.assertRaises(RuntimeError) as ctx:
                load_vip_vl_pretrained(
                    model_or_encoder=adapter,
                    checkpoint_path_or_repo=ckpt_path,
                )
            self.assertIn("Shape mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
