"""Random-access audio reader for uncompressed ZIP shards.

Notebook 01 stores prepared WAVs in a small number of ZIP_STORED shards so
Kaggle can preserve them as notebook output without thousands of individual
files.  Notebook 02 mounts those shards read-only under ``/kaggle/input`` and
uses this reader without extracting them into ``/kaggle/working``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import soundfile as sf


SHARD_INDEX_FILENAME = "audio_shards_index.json"


class ShardedZipAudioReader:
    """Read WAV members by relative path from verified ZIP_STORED shards."""

    def __init__(self, prepared_root: Union[str, Path]):
        self.prepared_root = Path(prepared_root)
        self.index_path = self.prepared_root / SHARD_INDEX_FILENAME
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Audio shard index not found: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as stream:
            index = json.load(stream)
        if index.get("schema_version") != 1 or index.get("storage") != "zip_stored_shards":
            raise RuntimeError(f"Unsupported audio shard index: {self.index_path}")

        members = index.get("members")
        if not isinstance(members, dict):
            raise RuntimeError(f"Invalid members table in {self.index_path}")
        self.members: Dict[str, Dict[str, Any]] = members
        self._archives: Dict[str, zipfile.ZipFile] = {}

    def __contains__(self, audio_relpath: str) -> bool:
        return Path(audio_relpath).as_posix() in self.members

    def _archive(self, relative_path: str) -> zipfile.ZipFile:
        archive = self._archives.get(relative_path)
        if archive is None:
            archive_path = self.prepared_root / relative_path
            archive = zipfile.ZipFile(archive_path, mode="r")
            self._archives[relative_path] = archive
        return archive

    def read(self, audio_relpath: str) -> Tuple[np.ndarray, int]:
        """Return one mono float32 waveform and its sample rate."""
        key = Path(audio_relpath).as_posix()
        record = self.members.get(key)
        if record is None:
            raise FileNotFoundError(f"Audio member not found in shard index: {key}")

        archive = self._archive(str(record["shard"]))
        member = str(record.get("member", key))
        with archive.open(member, mode="r") as source:
            payload = source.read()
        if len(payload) != int(record["size_bytes"]):
            raise IOError(
                f"Truncated audio member {member}: {len(payload)} != {record['size_bytes']} bytes"
            )

        waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return np.ascontiguousarray(waveform), int(sample_rate)

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __getstate__(self) -> Dict[str, Any]:
        """Do not pickle open ZIP handles when DataLoader uses spawn workers."""
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
