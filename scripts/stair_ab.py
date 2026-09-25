"""A/B probe: does the runner's force_button_edge reach the emulator?

Backs up the emulator just before the stair section during a live policy run,
then continues twice from the identical state: once with the policy untouched,
once forcing a release edge before every jump macro. Prints both trajectories.
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


def find_core(env):
    current = env
    for _ in range(10):
        if hasattr(current, "dump_state"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("no dump_state-capable env")


def play_window(env, info, action, force_edge, grounded, prev_action):
    stepped = 0
    terminated = False
    if action in JUMP_ACTIONS and (force_edge or (prev_action in JUMP_ACTIONS and grounded)):
        _, _, terminated, _, info = env.step(ACTION_TO_INDEX[JUMP_RELEASE_ACTION[action]])
        stepped += 1
    while stepped < 8 and not terminated:
        _, _, terminated, _, info = env.step(ACTION_TO_INDEX[action])
        stepped += 1
    return info, terminated


def run_from(env, parser, policy, info, decisions, force_all_edges=False, label=""):
    previous_action = None
    frames_since_parse = 0
    trace = []
    for _ in range(decisions):
        snapshot = parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=previous_action.value if previous_action else None,
            previous_response_delay_frames=0,
            elapsed_frames=frames_since_parse,
        )
        if snapshot.dead or snapshot.clear:
            break
        decision = policy.choose(snapshot, actions=tuple(Action))
        edge = decision.force_button_edge or (force_all_edges and decision.action in JUMP_ACTIONS)
        info, terminated = play_window(
            env, info, decision.action, edge, snapshot.grounded, previous_action
        )
        trace.append((snapshot.x, snapshot.y, decision.action.value, decision.force_button_edge))
        previous_action = decision.action
        frames_since_parse = 8
        if terminated:
            break
    print(f"{label}: end x={info['x_pos']}")
    for x, y, a, fe in trace:
        print(f"   x={x:4d} y={y:3d} {a:15s} force_edge={fe}")
    return info


def main() -> None:
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    core = find_core(env)
    _, info = env.reset(seed=123)
    parser = MarioStateParser(decision_horizon_frames=8)
    policy = LayaPolicy(checkpoint=ROOT / "artifacts" / "laya-mario-noul")

    # drive to just before the stair section with the real policy
    previous_action = None
    state_blob = None
    frames_since_parse = 0
    for _ in range(400):
        snapshot = parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=previous_action.value if previous_action else None,
            previous_response_delay_frames=0,
            elapsed_frames=frames_since_parse,
        )
        if snapshot.dead or snapshot.clear:
            break
        if 2130 <= snapshot.x <= 2170 and state_blob is None:
            state_blob = core.dump_state()
            print(f"backup at x={snapshot.x} y={snapshot.y}")
        decision = policy.choose(snapshot, actions=tuple(Action))
        info, terminated = play_window(
            env,
            info,
            decision.action,
            decision.force_button_edge,
            snapshot.grounded,
            previous_action,
        )
        previous_action = decision.action
        frames_since_parse = 8
        if terminated:
            break
    print(f"approach run reached x={info['x_pos']}")

    if state_blob is None:
        print("no backup")
        return

    for label, force_all in (("policy-as-is", False), ("force-edge-everywhere", True)):
        core.load_state(state_blob)
        _, _, _, _, info = env.step(ACTION_TO_INDEX[Action.NOOP])
        fresh_parser = MarioStateParser(decision_horizon_frames=8)
        fresh_parser.parse(info, _unwrap_ram(env))
        run_from(env, fresh_parser, policy, info, 18, force_all_edges=force_all, label=label)
    env.close()


if __name__ == "__main__":
    main()
