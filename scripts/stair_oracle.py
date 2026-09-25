"""Oracle search at the live stair-approach state.

Mirrors run_episode's headless loop exactly (release edges, landing break), runs
the real Laya policy until Mario is grounded in the staircase approach, backs up
the emulator state (nes_py _backup/_restore), lets the policy continue to its
death, then replays scripted suffixes from the backup to find what survives.

Usage: .venv/bin/python scripts/stair_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from typesafe_mario.actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from typesafe_mario.policy import LayaPolicy
from typesafe_mario.runner import _unwrap_ram, create_mario_env
from typesafe_mario.state import MarioStateParser


def find_backupable(env):
    current = env
    for _ in range(10):
        if hasattr(current, "dump_state"):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise RuntimeError("no dump_state-capable env found")


def step_window(env, info, action, previous_action, snapshot, frames_per_decision=8):
    """run_episode's exact per-window stepping (edge + enemy-gated landing break)."""
    stepped = 0
    terminated = truncated = False
    if action in JUMP_ACTIONS and (previous_action in JUMP_ACTIONS and snapshot.grounded):
        _, _, terminated, truncated, info = env.step(ACTION_TO_INDEX[JUMP_RELEASE_ACTION[action]])
        stepped += 1
    threat = snapshot.threat_features()
    break_on_landing = (
        snapshot.jump_phase == "falling"
        and threat["enemy_ahead"]
        and (threat["nearest_enemy_distance_pixels"] or 999) <= 48
    )
    prev_y = info["y_pos"]
    while stepped < frames_per_decision and not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
        stepped += 1
        if break_on_landing and info["y_pos"] >= prev_y:
            break
        prev_y = info["y_pos"]
    return info, terminated or truncated, stepped


def main() -> None:
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    core = find_backupable(env)
    _, info = env.reset(seed=123)
    parser = MarioStateParser(decision_horizon_frames=8)
    policy = LayaPolicy(checkpoint=ROOT / "artifacts" / "laya-mario-noul")

    actions = tuple(Action)
    previous_action = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    frames_since_parse = 0
    backups: list[tuple[int, object]] = []
    for decision_index in range(800):
        snapshot = parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=previous_action.value if previous_action else None,
            previous_reward=previous_reward,
            previous_latency_ms=previous_latency_ms,
            previous_response_delay_frames=0,
            elapsed_frames=frames_since_parse,
        )
        if snapshot.dead or snapshot.clear:
            break
        if 2240 <= snapshot.x <= 2255 and not backups:
            backups.append((snapshot.x, core.dump_state()))
            print(f"backup at x={snapshot.x} y={snapshot.y} grounded={snapshot.grounded}")
        decision = policy.choose(snapshot, actions)
        if decision.force_button_edge:
            print(f"FORCE EDGE at x={snapshot.x} y={snapshot.y} phase={snapshot.jump_phase}")
        object.__setattr__(snapshot, "_force_edge", decision.force_button_edge)
        info, done, frames_since_parse = step_window(
            env, info, decision.action, previous_action, snapshot
        )
        previous_action = decision.action
        previous_reward = 0.0
        previous_latency_ms = decision.latency_ms
        if done:
            break
    print(f"policy baseline ended at x={info['x_pos']}")

    if not backups:
        print("never reached the stair region; giving up")
        env.close()
        return

    suffixes = {
        "all-RRJ": [Action.RIGHT_RUN_JUMP] * 8,
        "run-then-hop": [Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP] * 4,
        "two-run-then-hop": [Action.RIGHT_RUN, Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP] * 3,
        "hop-run": [Action.RIGHT_RUN_JUMP, Action.RIGHT_RUN] * 4,
    }
    for name, suffix in suffixes.items():
        core.load_state(backups[0][1])
        _, _, _, _, info = env.step(ACTION_TO_INDEX[Action.NOOP])
        prev = None
        terminated = False
        for a in suffix:
            if a in JUMP_ACTIONS and prev in JUMP_ACTIONS:
                env.step(ACTION_TO_INDEX[JUMP_RELEASE_ACTION[a]])
            for _ in range(8):
                _, _, terminated, _, info = env.step(ACTION_TO_INDEX[a])
                if terminated:
                    break
            prev = a
            if terminated:
                break
        if not terminated:
            for _ in range(400):
                _, _, terminated, _, info = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN])
                if terminated or info["x_pos"] > 2620:
                    break
        print(f"{name}: x={info['x_pos']} {'DIED' if terminated else 'alive'}")
    env.close()


if __name__ == "__main__":
    main()
