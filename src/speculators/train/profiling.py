"""Lightweight step-level profiling for the training loop."""

import threading
import time
from collections import defaultdict

import torch

PHASES = ("fetch", "h2d", "fwd", "bwd", "opt")


class StepTimer:
    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.steps_seen = 0
        self._sustained_time = 0.0
        self._sustained_fetch = 0.0
        self._sustained_tokens = 0
        self._sustained_steps = 0
        self._reset_window()

    def _reset_window(self):
        self._win_sum: dict[str, float] = defaultdict(float)
        self._win_steps = 0
        self._win_tokens = 0

    def record(self, phases: dict[str, float], tokens: int) -> None:
        self.steps_seen += 1
        for k in PHASES:
            self._win_sum[k] += phases.get(k, 0.0)
        self._win_steps += 1
        self._win_tokens += tokens

        if self.steps_seen > self.warmup_steps:
            step_time = sum(phases.get(k, 0.0) for k in PHASES)
            self._sustained_time += step_time
            self._sustained_fetch += phases.get("fetch", 0.0)
            self._sustained_tokens += tokens
            self._sustained_steps += 1

    def window_snapshot(self) -> dict[str, float]:
        if self._win_steps == 0:
            return {}
        means_s = {k: self._win_sum[k] / self._win_steps for k in PHASES}
        step_s = sum(means_s.values())
        snap: dict[str, float] = {f"{k}_ms": means_s[k] * 1000 for k in PHASES}
        snap["step_ms"] = step_s * 1000
        win_wall = sum(self._win_sum.values())
        snap["tokens_per_s"] = self._win_tokens / win_wall if win_wall > 0 else 0.0
        snap["fetch_frac"] = self._win_sum["fetch"] / win_wall if win_wall > 0 else 0.0
        self._reset_window()
        return snap

    def sustained_snapshot(self) -> dict[str, float]:
        if self._sustained_steps == 0:
            return {}
        return {
            "sustained_tokens_per_s": self._sustained_tokens / self._sustained_time,
            "sustained_fetch_frac": self._sustained_fetch / self._sustained_time,
            "sustained_step_ms": (self._sustained_time / self._sustained_steps) * 1000,
            "sustained_steps": float(self._sustained_steps),
        }


class GpuUtilSampler:
    def __init__(self, device: int, interval_s: float = 0.25):
        self.device = device
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sum = 0.0
        self._count = 0
        self._enabled = False

    def start(self) -> None:
        try:
            torch.cuda.utilization(self.device)
        except Exception:
            return
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                u = float(torch.cuda.utilization(self.device))
            except Exception:
                continue
            with self._lock:
                self._sum += u
                self._count += 1

    def pop_mean(self) -> float | None:
        if not self._enabled:
            return None
        with self._lock:
            if self._count == 0:
                return None
            mean = self._sum / self._count
            self._sum = 0.0
            self._count = 0
            return mean
