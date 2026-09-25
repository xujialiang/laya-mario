"""Run a dashboard episode and record it to an MP4 in one go.

Patches ``LiveDashboard.draw`` so every rendered frame is piped straight
into ffmpeg, and quits automatically a few seconds after the run ends so
an unattended recording finishes by itself (the dashboard otherwise keeps
waiting for a manual QUIT after death or level completion).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from typesafe_mario.dashboard import DashboardCommand, LiveDashboard
from typesafe_mario.policy import FinishingMacroPolicy, LayaPolicy
from typesafe_mario.runner import run_episode


class DashboardRecorder:
    """Streams dashboard frames into an ffmpeg libx264 pipe."""

    def __init__(self, output: Path, width: int, height: int, fps: int) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.frames = 0

    def write(self, screen: pygame.Surface) -> None:
        pixels = np.ascontiguousarray(
            pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        )
        assert self.ffmpeg.stdin is not None
        self.ffmpeg.stdin.write(pixels.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.ffmpeg.stdin is not None:
            self.ffmpeg.stdin.close()
        code = self.ffmpeg.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code} after {self.frames} frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dashboard-run.mp4"))
    parser.add_argument("--laya-checkpoint", type=Path, default=None)
    parser.add_argument("--finish-macro", type=Path, default=None)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--end-hold-seconds",
        type=float,
        default=8.0,
        help="How long to keep recording the ended dashboard before quitting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    original_draw = LiveDashboard.draw
    state: dict[str, object] = {"recorder": None, "ended_at": None}

    def draw_and_record(
        self: LiveDashboard,
        frame: object,
        snapshot: object,
        decision: object,
        **kwargs: object,
    ) -> DashboardCommand:
        command = original_draw(self, frame, snapshot, decision, **kwargs)
        recorder = state["recorder"]
        if recorder is None:
            recorder = DashboardRecorder(
                args.output, self.config.width, self.config.height, self.config.fps
            )
            state["recorder"] = recorder
        recorder.write(self.screen)
        if kwargs.get("run_ended"):
            ended_at = state["ended_at"]
            if ended_at is None:
                state["ended_at"] = time.monotonic()
            elif time.monotonic() - float(ended_at) > args.end_hold_seconds:
                command = DashboardCommand.QUIT
        else:
            state["ended_at"] = None
        return command

    LiveDashboard.draw = draw_and_record  # type: ignore[method-assign]

    policy = LayaPolicy(checkpoint=args.laya_checkpoint)
    if args.finish_macro is not None:
        policy = FinishingMacroPolicy(policy, args.finish_macro)

    started = time.monotonic()
    try:
        log_path = run_episode(
            env_id="SuperMarioBros-1-1-v0",
            policy=policy,
            frames_per_decision=8,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=Path("artifacts"),
            display="dashboard",
        )
    finally:
        recorder = state["recorder"]
        if recorder is not None:
            recorder.close()
            elapsed = time.monotonic() - started
            print(
                f"Recorded {recorder.frames} frames ({elapsed:.0f}s) "
                f"to {args.output.resolve()}"
            )
    print(f"Run log: {log_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
