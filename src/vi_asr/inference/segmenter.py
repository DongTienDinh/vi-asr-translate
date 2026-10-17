"""Voice Activity Detection (VAD) and long-audio segmenter (Section 26)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioSegment:
    """Represents a speech segment extracted from long audio."""
    segment_id: int
    start_sec: float
    end_sec: float
    waveform: torch.Tensor  # [1, num_samples] at 16kHz

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class EnergyVADSegmenter:
    """Energy-based VAD segmenter for splitting long audio into bounded segments."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        energy_threshold: float = 1e-4,
        min_speech_duration_sec: float = 1.0,
        max_segment_duration_sec: float = 12.0,
        min_silence_duration_sec: float = 0.4,
    ):
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_length_ms / 1000.0)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000.0)
        self.energy_threshold = energy_threshold
        self.min_speech_sec = min_speech_duration_sec
        self.max_segment_sec = max_segment_duration_sec
        self.min_silence_sec = min_silence_duration_sec

    def segment_waveform(self, waveform: torch.Tensor) -> List[AudioSegment]:
        """Segment a 1D or 2D [1, T] 16kHz waveform into speech chunks."""
        if waveform.dim() == 2:
            wav_1d = waveform.squeeze(0)
        else:
            wav_1d = waveform

        total_samples = wav_1d.size(0)
        total_duration = total_samples / float(self.sample_rate)

        if total_duration <= self.max_segment_sec:
            # Utterance is already short enough
            return [
                AudioSegment(
                    segment_id=0,
                    start_sec=0.0,
                    end_sec=total_duration,
                    waveform=wav_1d.unsqueeze(0),
                )
            ]

        # Compute frame energy
        num_frames = max(1, (total_samples - self.frame_len) // self.frame_shift + 1)
        energies = []
        for i in range(num_frames):
            start = i * self.frame_shift
            end = start + self.frame_len
            frame = wav_1d[start:end]
            e = torch.mean(frame**2).item()
            energies.append(e)

        is_speech = [e > self.energy_threshold for e in energies]

        # Find candidate silence split points
        silence_frames_threshold = int(self.min_silence_sec * self.sample_rate / self.frame_shift)
        segments: List[AudioSegment] = []

        cur_start_frame = 0
        cur_frame = 0
        seg_id = 0

        max_frames_per_seg = int(self.max_segment_sec * self.sample_rate / self.frame_shift)
        min_frames_per_seg = int(self.min_speech_sec * self.sample_rate / self.frame_shift)

        while cur_frame < num_frames:
            seg_len = cur_frame - cur_start_frame

            # Check if current frame is silence and segment is long enough
            is_silence = not is_speech[cur_frame] if cur_frame < len(is_speech) else True
            reached_max = seg_len >= max_frames_per_seg

            if (is_silence and seg_len >= min_frames_per_seg) or reached_max:
                start_sample = cur_start_frame * self.frame_shift
                end_sample = min(total_samples, cur_frame * self.frame_shift + self.frame_len)

                seg_wav = wav_1d[start_sample:end_sample].unsqueeze(0)
                segments.append(
                    AudioSegment(
                        segment_id=seg_id,
                        start_sec=start_sample / float(self.sample_rate),
                        end_sec=end_sample / float(self.sample_rate),
                        waveform=seg_wav,
                    )
                )
                seg_id += 1
                cur_start_frame = cur_frame + 1

            cur_frame += 1

        # Final trailing segment
        if cur_start_frame < num_frames:
            start_sample = cur_start_frame * self.frame_shift
            seg_wav = wav_1d[start_sample:].unsqueeze(0)
            if seg_wav.size(1) >= self.frame_len:
                segments.append(
                    AudioSegment(
                        segment_id=seg_id,
                        start_sec=start_sample / float(self.sample_rate),
                        end_sec=total_samples / float(self.sample_rate),
                        waveform=seg_wav,
                    )
                )

        return segments if segments else [
            AudioSegment(
                segment_id=0,
                start_sec=0.0,
                end_sec=total_duration,
                waveform=wav_1d.unsqueeze(0),
            )
        ]

    def segment_file(self, audio_path: Union[str, Path]) -> List[AudioSegment]:
        """Read audio file, resample to 16kHz mono if needed, and segment."""
        wav_np, sr = sf.read(str(audio_path), dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        wav_t = torch.from_numpy(wav_np).float()
        if sr != self.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, self.sample_rate)
            wav_t = resampler(wav_t)

        return self.segment_waveform(wav_t)
