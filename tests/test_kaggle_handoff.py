import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.kaggle_handoff import build_handoff, restore_handoff, validate_audio_shards
from vi_asr.data.audio_store import ShardedZipAudioReader


class KaggleHandoffTest(unittest.TestCase):
    def test_build_and_restore_verified_handoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            artifacts = source / "artifacts"
            prepared = artifacts / "prepared"
            teacher = artifacts / "teacher_cache"
            tokenizer = source / "project_tokenizer"
            audit = artifacts / "audit_report"

            sample_ids = {}
            for split in ("train", "valid", "test"):
                sample_id = f"{split}-001"
                sample_ids[split] = sample_id
                relpath = f"audio/{split}/{sample_id}.wav"
                audio = prepared / relpath
                audio.parent.mkdir(parents=True, exist_ok=True)
                sf.write(audio, np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000, subtype="PCM_16")
                row = {
                    "sample_id": sample_id,
                    "audio_relpath": relpath,
                    "text_asr": "xin chao",
                }
                prepared.mkdir(parents=True, exist_ok=True)
                (prepared / f"{split}.jsonl").write_text(
                    json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
                )

            teacher.mkdir(parents=True)
            (teacher / "meta.json").write_text(
                json.dumps(
                    {
                        "total_samples": 1,
                        "total_tokens": 2,
                        "hidden_size": 768,
                        "dtype": "float16",
                    }
                ),
                encoding="utf-8",
            )
            (teacher / "index.jsonl").write_text(
                json.dumps({"sample_id": sample_ids["train"], "start_offset": 0, "num_tokens": 2})
                + "\n",
                encoding="utf-8",
            )
            np.save(teacher / "tokens-000.npy", np.zeros((2, 768), dtype=np.float16))
            np.save(teacher / "token_ids-000.npy", np.zeros((2,), dtype=np.int32))
            np.save(teacher / "offsets-000.npy", np.array([0, 2], dtype=np.int64))

            tokenizer.mkdir(parents=True)
            (tokenizer / "bpe_1024.model").write_bytes(b"model")
            (tokenizer / "bpe_1024.vocab").write_text("<pad>\t0\n", encoding="utf-8")
            audit.mkdir(parents=True)
            (audit / "audit.json").write_text("{}", encoding="utf-8")

            built = build_handoff(
                argparse.Namespace(
                    prepared_root=str(prepared),
                    teacher_root=str(teacher),
                    tokenizer_root=str(tokenizer),
                    audit_root=str(audit),
                    artifact_root=str(artifacts),
                    dataset_revision="test-revision",
                    remove_source_audio=True,
                    shard_size_mib=1,
                )
            )
            self.assertEqual(built["splits"]["train"]["samples"], 1)
            self.assertEqual(built["audio_storage"]["files"], 3)
            self.assertFalse(any(prepared.rglob("*.wav")))
            self.assertEqual(validate_audio_shards(prepared)["files"], 3)

            working = root / "working"
            restored = restore_handoff(
                argparse.Namespace(
                    input_root=str(source),
                    working_root=str(working),
                    runtime_paths="configs/paths/kaggle_runtime.yaml",
                    skip_archive_hash=False,
                )
            )
            self.assertTrue(restored["audio_shards_sha256_verified"])
            self.assertEqual(restored["audio_storage"]["files"], 3)
            reader = ShardedZipAudioReader(prepared)
            waveform, sample_rate = reader.read("audio/train/train-001.wav")
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(waveform.shape, (1600,))
            reader.close()
            runtime = working / "vi_asr_project" / "configs" / "paths" / "kaggle_runtime.yaml"
            self.assertTrue(runtime.is_file())
            runtime_data = json.loads(runtime.read_text(encoding="utf-8"))
            self.assertEqual(Path(runtime_data["paths"]["data_root"]), prepared)
            self.assertTrue(
                (working / "vi_asr_project" / "artifacts" / "tokenizer" / "bpe_1024.model").is_file()
            )


if __name__ == "__main__":
    unittest.main()
