import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from event_lads.LADS import LADS


def _build_events(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    t = data["t"].astype(np.float32) / 1e6
    x = data["x"].astype(np.int64)
    y = data["y"].astype(np.int64)
    p = data["p"].astype(np.int64)
    return np.stack([t, x, y, p], axis=1)


def _split_events(events: np.ndarray, frames_number: int, split_by: str) -> List[np.ndarray]:
    if frames_number <= 0:
        raise ValueError(f"frames_number must be positive, got {frames_number}.")

    if len(events) == 0:
        return [events.copy() for _ in range(frames_number)]

    if split_by == "number":
        return [chunk.astype(np.float32, copy=False) for chunk in np.array_split(events, frames_number)]

    if split_by == "time":
        start_t = events[0, 0]
        end_t = events[-1, 0]
        if end_t <= start_t:
            return [events.copy() if i == 0 else events[:0].copy() for i in range(frames_number)]

        edges = np.linspace(start_t, end_t, frames_number + 1, dtype=np.float32)
        chunks = []
        for i in range(frames_number):
            left = edges[i]
            right = edges[i + 1]
            if i == frames_number - 1:
                mask = (events[:, 0] >= left) & (events[:, 0] <= right)
            else:
                mask = (events[:, 0] >= left) & (events[:, 0] < right)
            chunks.append(events[mask].astype(np.float32, copy=False))
        return chunks

    raise ValueError(f"Unsupported split_by: {split_by}. Expected 'number' or 'time'.")


def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
    max_abs = tensor.abs().amax()
    if torch.isfinite(max_abs) and max_abs > 0:
        tensor = tensor / max_abs
    return tensor.to(torch.float32)


def _events_to_raw_frame(chunk: np.ndarray, sensor_size: Tuple[int, int]) -> torch.Tensor:
    height, width = sensor_size
    raw = np.zeros((2, height, width), dtype=np.float32)

    if len(chunk) == 0:
        return torch.from_numpy(raw)

    xs = np.clip(chunk[:, 1].astype(np.int64), 0, width - 1)
    ys = np.clip(chunk[:, 2].astype(np.int64), 0, height - 1)
    ps = chunk[:, 3] > 0

    np.add.at(raw[0], (ys[ps], xs[ps]), 1.0)
    np.add.at(raw[1], (ys[~ps], xs[~ps]), 1.0)

    raw_tensor = torch.from_numpy(raw)
    return _normalize_map(raw_tensor)


def events_to_lads_hybrid_frames(
    events: np.ndarray,
    frames_number: int,
    sensor_size=(128, 128),
    split_by: str = "number",
    lads_kwargs: Dict = None,
) -> torch.Tensor:
    lads_kwargs = dict(lads_kwargs or {})
    height, width = sensor_size
    processor = LADS(
        H=height,
        W=width,
        device=lads_kwargs.pop("device", "cpu"),
        ts_to_seconds_factor=lads_kwargs.pop("ts_to_seconds_factor", 1.0),
        **lads_kwargs,
    )

    frames = []
    for chunk in _split_events(events, frames_number=frames_number, split_by=split_by):
        raw_frame = _events_to_raw_frame(chunk, sensor_size=sensor_size)
        surface, _, _ = processor.integrateEvents(chunk)
        surface = _normalize_map(surface).unsqueeze(0)
        frames.append(torch.cat([raw_frame, surface], dim=0))

    return torch.stack(frames, dim=0).to(torch.float32)


class LADSDVS128Gesture(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        frames_number: int = 16,
        split_by: str = "number",
        sensor_size=(128, 128),
        lads_kwargs: Dict = None,
    ):
        super().__init__()
        split = "train" if train else "test"
        self.root = root
        self.events_root = os.path.join(root, "events_np", split)
        self.frames_number = frames_number
        self.split_by = split_by
        self.sensor_size = sensor_size
        self.lads_kwargs = dict(lads_kwargs or {})

        if not os.path.isdir(self.events_root):
            raise FileNotFoundError(f"Cannot find events directory: {self.events_root}")

        self.samples = []
        for label_name in sorted(os.listdir(self.events_root), key=lambda x: int(x)):
            label_dir = os.path.join(self.events_root, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = int(label_name)
            for file_name in sorted(os.listdir(label_dir)):
                if file_name.endswith(".npz"):
                    self.samples.append((os.path.join(label_dir, file_name), label))

        if not self.samples:
            raise RuntimeError(f"No event samples were found under {self.events_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_path, label = self.samples[index]
        events = _build_events(file_path)
        frames = events_to_lads_hybrid_frames(
            events,
            frames_number=self.frames_number,
            sensor_size=self.sensor_size,
            split_by=self.split_by,
            lads_kwargs=self.lads_kwargs,
        )
        return frames, label
