"""Faithful replay of a logged headless run, then try alternative action suffixes.

The headless runner's window logic (release edge, enemy-gated landing break) is
replicated exactly, so a replay of the logged actions reproduces the logged death;
from any decision point we can then test what suffix would have survived.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from typesafe_mario.actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from typesafe_mario.runner import create_mario_env


def run_window(env, info, action, prev_action, state, frames_per_decision=8):
    """One decision window, mirroring run_episode's headless loop."""
    stepped = 0
    terminated = False
    threat = state["hazard"]
    break_on_landing = (
        state["player"]["jump_phase"] == "falling"
        and threat["enemy_ahead"]
        and (threat["nearest_enemy_distance_pixels"] or 999) <= 48
    )
    if action in JUMP_ACTIONS and prev_action in JUMP_ACTIONS and state["player"]["grounded"]:
        _, _, terminated, _, info = env.step(ACTION_TO_INDEX[JUMP_RELEASE_ACTION[action]])
        stepped += 1
    prev_y = info["y_pos"]
    while stepped < frames_per_decision and not terminated:
        _, _, terminated, _, info = env.step(ACTION_TO_INDEX[action])
        stepped += 1
        if break_on_landing and info["y_pos"] >= prev_y:
            break
        prev_y = info["y_pos"]
    return info, terminated


def replay(rows, n, seed=123):
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    _, info = env.reset(seed=seed)
    prev = None
    terminated = False
    for r in rows[:n]:
        a = Action(r["action"])
        info, terminated = run_window(env, info, a, prev, r["state"])
        if terminated:
            break
        prev = a
    return env, info, terminated, prev


def play_out(env, info, prev, actions):
    """Continue with a scripted action list (one per 8-frame window, plain rules)."""
    for a in actions:
        if a in JUMP_ACTIONS and prev in JUMP_ACTIONS:
            env.step(ACTION_TO_INDEX[JUMP_RELEASE_ACTION[a]])
        terminated = False
        for _ in range(8):
            _, _, terminated, _, info = env.step(ACTION_TO_INDEX[a])
            if terminated:
                break
        prev = a
        if terminated:
            break
    return info, terminated


def main() -> None:
    run_path, cut = sys.argv[1], int(sys.argv[2])
    suffixes = json.loads(sys.argv[3])  # list of action-name lists
    with Path(run_path).open() as log_file:
        rows = [json.loads(line) for line in log_file]

    env, info, terminated, _ = replay(rows, len(rows))
    print(
        f"replay check: x={info['x_pos']} term={terminated} (log ends x={rows[-1]['state']['player']['x']})"
    )
    env.close()

    for suffix in suffixes:
        env, info, terminated, prev = replay(rows, cut)
        start_x = info["x_pos"]
        info, terminated = play_out(env, info, prev, [Action(a) for a in suffix])
        print(
            f"cut@{cut} x={start_x} + {suffix[:6]}{'...' if len(suffix) > 6 else ''} "
            f"-> x={info['x_pos']} {'DIED' if terminated else 'alive'}"
        )
        env.close()


if __name__ == "__main__":
    main()
