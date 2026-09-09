"""
Utilities for reading the BrainHack Vanderbilt 2026 sEEG datasets.

The datasets are MATLAB v7.3 (HDF5) files and are read with h5py rather than
scipy.io.loadmat. The module gives the Spatial and Shape tasks a common
interface (`SEEGDataset` / `ChannelView`) and loads trial signals lazily, so
opening a multi-GB file is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd

SAMPLING_RATE_HZ: float = 512.0
REFERENCE_SCHEMES: tuple[str, ...] = ("TrialData", "Common", "Laplacian")


# HDF5 helpers
def _to_str(dataset: h5py.Dataset) -> str:
    """Decode a MATLAB char array (stored as uint16) into a Python str."""
    arr = np.asarray(dataset)
    if arr.dtype.kind in ("u", "i"):
        return "".join(chr(int(c)) for c in arr.flatten())
    return str(arr)


def _scalar(dataset: h5py.Dataset) -> float:
    return float(np.asarray(dataset).flatten()[0])


class _Deref:
    """Resolve the HDF5 object-reference indirection used by MATLAB v7.3."""
    def __init__(self, file: h5py.File) -> None:
        self._file = file

    def string(self, ref) -> str:
        obj = self._file[ref]
        # cell array of char -> one more hop
        if obj.dtype == object:
            return _to_str(self._file[obj[0, 0]])
        return _to_str(obj)

    def array(self, ref) -> np.ndarray:
        return np.asarray(self._file[ref])


@dataclass(frozen=True)
class Trial:
    """A single trial's labels (time series are fetched lazily via ChannelView)."""
    index: int
    outcome: str  # "correct" or "error"
    cls: int
    cue_shape: float | None = None
    match_shape: float | None = None
    is_match: float | None = None


class ChannelView:
    """One entry of the SEEG struct array; trial signals load on demand."""

    def __init__(self, dataset: "SEEGDataset", idx: int) -> None:
        self._ds = dataset
        self._idx = idx
        f = dataset._file
        seeg = f["SEEG"]
        d = dataset._deref

        self.index: int = idx
        self.sub: str = d.string(seeg["Sub"][idx, 0])
        self.task: str = d.string(seeg["Task"][idx, 0])
        self.condition: str = d.string(seeg["Condition"][idx, 0])
        self.channel_id: int = int(_scalar(f[seeg["Channel"][idx, 0]]))
        self.channel_label: str = d.string(seeg["Channel_Label"][idx, 0])
        self.subregion: str = d.string(seeg["subRegion"][idx, 0])
        # `Prefrontal_subdiv` has an inconsistent-casing bug in the source data
        # ('dorsal' vs 'Dorsal'); normalise to Title case.
        self.prefrontal_subdiv: str = d.string(
            seeg["Prefrontal_subdiv"][idx, 0]
        ).strip().title()
        self.hemisphere: str = d.string(seeg["Hemisphere"][idx, 0])

        self._correct_grp = f[seeg[dataset._correct_field][idx, 0]]
        self._error_grp = (
            f[seeg["Error"][idx, 0]] if dataset._has_error else None
        )

    @property
    def n_correct(self) -> int:
        return self._grp_len(self._correct_grp)

    @property
    def n_error(self) -> int:
        return self._grp_len(self._error_grp) if self._error_grp is not None else 0

    @staticmethod
    def _grp_len(grp) -> int:
        if grp is None or "Class" not in grp:
            return 0
        return int(grp["Class"].shape[0])

    def trials(self, outcome: str = "correct") -> list[Trial]:
        grp = self._pick(outcome)
        out: list[Trial] = []
        for k in range(self._grp_len(grp)):
            out.append(
                Trial(
                    index=k,
                    outcome=outcome,
                    cls=int(_scalar(self._ds._file[grp["Class"][k, 0]])),
                    cue_shape=self._opt(grp, "CueShape", k),
                    match_shape=self._opt(grp, "MatchShape", k),
                    is_match=self._opt(grp, "IsMatch", k),
                )
            )
        return out

    def classes(self, outcome: str = "correct") -> np.ndarray:
        grp = self._pick(outcome)
        return np.array(
            [
                int(_scalar(self._ds._file[grp["Class"][k, 0]]))
                for k in range(self._grp_len(grp))
            ],
            dtype=int,
        )

    def signal(
        self, trial_idx: int, ref: str = "TrialData", outcome: str = "correct"
    ) -> np.ndarray:
        """Voltage trace (n_samples,) for one trial and reference."""
        if ref not in REFERENCE_SCHEMES:
            raise ValueError(f"ref must be one of {REFERENCE_SCHEMES}, got {ref!r}")
        grp = self._pick(outcome)
        obj = self._ds._file[grp[ref][trial_idx, 0]]
        if obj.attrs.get("MATLAB_empty", 0):
            return np.empty((0,), dtype=np.float64)
        return np.asarray(obj).flatten().astype(np.float64)

    def has_signal(
        self, trial_idx: int, ref: str = "TrialData", outcome: str = "correct"
    ) -> bool:
        grp = self._pick(outcome)
        obj = self._ds._file[grp[ref][trial_idx, 0]]
        return not obj.attrs.get("MATLAB_empty", 0) and np.asarray(obj).size > 1

    def matrix(
        self, ref: str = "TrialData", outcome: str = "correct", drop_empty: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack trials into (n_trials, n_samples) with matching class labels."""
        grp = self._pick(outcome)
        n = self._grp_len(grp)
        if n == 0:
            return np.empty((0, 0)), np.empty((0,), dtype=int)
        labels = self.classes(outcome=outcome)
        rows, keep = [], []
        for k in range(n):
            s = self.signal(k, ref=ref, outcome=outcome)
            if s.size <= 1:
                if not drop_empty:
                    raise ValueError(
                        f"channel {self.index} trial {k} has no '{ref}' signal"
                    )
                continue
            rows.append(s)
            keep.append(k)
        if not rows:
            return np.empty((0, 0)), np.empty((0,), dtype=int)
        return np.vstack(rows), labels[keep]

    # internal helpers
    def _pick(self, outcome: str):
        if outcome == "correct":
            return self._correct_grp
        if outcome == "error":
            if self._error_grp is None:
                raise ValueError("this dataset has no Error trials")
            return self._error_grp
        raise ValueError("outcome must be 'correct' or 'error'")

    def _opt(self, grp, field: str, k: int) -> float | None:
        if field not in grp:
            return None
        return _scalar(self._ds._file[grp[field][k, 0]])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<ChannelView #{self.index} {self.sub} {self.channel_label} "
            f"{self.prefrontal_subdiv}/{self.subregion} "
            f"n_correct={self.n_correct}>"
        )


class SEEGDataset:
    """Open a v7.3 sEEG `.mat` file and iterate its channels lazily."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.path}'. Download it from Box and "
                f"place it in the data/ folder (see README)."
            )
        self._file = h5py.File(self.path, "r")
        seeg = self._file["SEEG"]
        self._deref = _Deref(self._file)
        self._correct_field = (
            "CorrectTrials" if "CorrectTrials" in seeg else "Correct"
        )
        self._has_error = "Error" in seeg
        self._n = int(seeg[self._correct_field].shape[0])
        self.fs = SAMPLING_RATE_HZ

    def __len__(self) -> int:
        return self._n

    def channel(self, idx: int) -> ChannelView:
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        return ChannelView(self, idx)

    def __getitem__(self, idx: int) -> ChannelView:
        return self.channel(idx)

    def __iter__(self) -> Iterator[ChannelView]:
        for i in range(self._n):
            yield self.channel(i)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SEEGDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def task(self) -> str:
        return self.channel(0).task

    @property
    def has_error_trials(self) -> bool:
        return self._has_error

    def time_vector(self, n_samples: int) -> np.ndarray:
        return np.arange(n_samples) / self.fs

    def summary(self) -> pd.DataFrame:
        """One row per channel: metadata + correct-trial counts."""
        rows = []
        for ch in self:
            cls = ch.classes()
            per_class = pd.Series(cls).value_counts().sort_index()
            rows.append(
                {
                    "idx": ch.index,
                    "sub": ch.sub,
                    "task": ch.task,
                    "condition": ch.condition,
                    "channel_id": ch.channel_id,
                    "channel_label": ch.channel_label,
                    "subregion": ch.subregion,
                    "prefrontal_subdiv": ch.prefrontal_subdiv,
                    "hemisphere": ch.hemisphere,
                    "n_correct": ch.n_correct,
                    "n_error": ch.n_error,
                    "n_classes": int(per_class.size),
                    "min_trials_per_class": int(per_class.min()) if per_class.size else 0,
                }
            )
        return pd.DataFrame(rows)

    def viable_channels(
        self, min_trials_per_class: int = 3, required_classes: int | None = None
    ) -> list[int]:
        """Channel indices with at least `min_trials_per_class` correct trials in
        every class present (and, if `required_classes` is given, covering that
        many distinct classes)."""
        keep: list[int] = []
        for ch in self:
            cls = ch.classes()
            if cls.size == 0:
                continue
            counts = pd.Series(cls).value_counts()
            if counts.min() < min_trials_per_class:
                continue
            if required_classes is not None and counts.size < required_classes:
                continue
            keep.append(ch.index)
        return keep
