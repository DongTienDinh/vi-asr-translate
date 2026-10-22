import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vi_asr.data.prepare import decode_audio_value, parse_sample_to_manifest_item


class _DecodedSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class _AudioDecoder:
    def __init__(self, data, sample_rate=16000):
        self._samples = _DecodedSamples(data, sample_rate)

    def get_all_samples(self):
        return self._samples


class PrepareAudioDecoderTest(unittest.TestCase):
    def test_modern_torchcodec_decoder_is_downmixed(self):
        decoder = _AudioDecoder(
            torch.stack(
                [torch.ones(1600, dtype=torch.float32), torch.zeros(1600, dtype=torch.float32)]
            )
        )
        waveform, sample_rate = decode_audio_value(decoder)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(waveform.shape, (1600,))
        np.testing.assert_allclose(waveform, 0.5, atol=1e-6)

    def test_parse_materializes_decoder_audio_to_wav(self):
        decoder = _AudioDecoder(torch.linspace(-0.2, 0.2, 3200).unsqueeze(0))
        sample = {"id": "train-001", "audio": decoder, "transcript": "Xin chào"}
        with tempfile.TemporaryDirectory() as temp_dir:
            item = parse_sample_to_manifest_item(
                sample,
                split="train",
                source_revision="test",
                sample_index=0,
                output_dir=temp_dir,
            )
            self.assertIsNotNone(item)
            self.assertEqual(item.num_samples, 3200)
            wav_path = Path(temp_dir) / "audio" / "train" / "train-001.wav"
            self.assertTrue(wav_path.is_file())
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(audio.shape, (3200,))

    def test_legacy_dictionary_contract_still_supported(self):
        waveform, sample_rate = decode_audio_value(
            {"array": np.zeros(800, dtype=np.float32), "sampling_rate": 8000}
        )
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(waveform.shape, (800,))


if __name__ == "__main__":
    unittest.main()
