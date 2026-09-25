"""Search a verified finishing macro for World 1-1's final gauntlet.

The policy plays headless until Mario reaches the gauntlet (x >= START_X), where
the emulator state (nes_py dump_state/load_state) and parser state are captured.
A beam search over decision windows - driven by the runner's own
step_decision_window, so semantics match live play exactly - looks for any
(action, edge) sequence that reaches the flagpole. The result is stored as a
position-triggered macro that FinishingMacroPolicy replays.

Usage: .venv/bin/python scripts/beam_finish.py
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from typesafe_mario.actions import Action
from typesafe_mario.policy import LayaPolicy
from typesafe_mario.runner import _unwrap_ram, create_mario_env, step_decision_window
from typesafe_mario.state import MarioStateParser

START_X = 2080
GOAL_X = 3175
BEAM_WIDTH = 48
MAX_DEPTH = 220
OUT_PATH = ROOT / "artifacts" / "finish-macro-1-1.json"

# Non-jump actions have no edge variant; jump actions get both.
BRANCHES = [(a, False) for a in (Action.RIGHT_RUN, Action.RIGHT, Action.NOOP, Action.LEFT)] + [
    (a, e) for a in (Action.RIGHT_RUN_JUMP, Action.RIGHT_JUMP, Action.JUMP) for e in (False, True)
]


def find_core(env):
    current = env
    for _ in range(10):
        if hasattr(current, "dump_state"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("no dump_state-capable env")


@dataclass
class Node:
    blob: bytes
    info: dict
    parser: MarioStateParser
    climb_phase: str | None
    stair_active: bool
    prev_action: Action | None
    x: int
    y: int
    history: tuple  # of (action_value, edge)


def play_prefix(env, policy, parser, until_x):
    """Drive the real policy with run_episode's exact headless semantics."""
    _, info = env.reset(seed=123)
    previous_action = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    frames_since_parse = 0
    for _ in range(800):
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
        if snapshot.x >= until_x:
            return snapshot, info, previous_action
        decision = policy.choose(snapshot, tuple(Action))
        _, info, terminated, _, stepped, _, total_reward = step_decision_window(
            env,
            info,
            action=decision.action,
            force_edge=decision.force_button_edge,
            previous_action=previous_action,
            snapshot=snapshot,
            frames_per_decision=8,
        )
        previous_action = decision.action
        previous_reward = total_reward
        previous_latency_ms = decision.latency_ms
        frames_since_parse = stepped
        if terminated:
            break
    return None, info, previous_action


def main() -> None:
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    core = find_core(env)
    parser = MarioStateParser(decision_horizon_frames=8)
    policy = LayaPolicy(checkpoint=ROOT / "artifacts" / "laya-mario-noul")

    snapshot, info, prev_action = play_prefix(env, policy, parser, START_X)
    if snapshot is None:
        print(f"policy never reached x={START_X}; died first. Fix the prefix first.")
        env.close()
        sys.exit(1)
    print(f"prefix reached x={snapshot.x} y={snapshot.y}; starting beam search")

    root = Node(
        blob=core.dump_state(),
        info=dict(info),
        parser=copy.deepcopy(parser),
        climb_phase=policy._climb_phase,
        stair_active=policy._stair_pair_active,
        prev_action=prev_action,
        x=snapshot.x,
        y=snapshot.y,
        history=(),
    )

    beam = [root]
    winner = None
    for depth in range(MAX_DEPTH):
        candidates: list[Node] = []
        for node in beam:
            for action, edge in BRANCHES:
                core.load_state(node.blob)
                node_parser = copy.deepcopy(node.parser)
                snap = node_parser.parse(
                    dict(node.info),
                    _unwrap_ram(env),
                    previous_action=node.prev_action.value if node.prev_action else None,
                    previous_response_delay_frames=0,
                    elapsed_frames=8,
                )
                _, child_info, terminated, _, _, _, _ = step_decision_window(
                    env,
                    dict(node.info),
                    action=action,
                    force_edge=edge,
                    previous_action=node.prev_action,
                    snapshot=snap,
                    frames_per_decision=8,
                )
                if terminated and not child_info.get("flag_get"):
                    continue
                candidates.append(
                    Node(
                        blob=core.dump_state(),
                        info=dict(child_info),
                        parser=node_parser,
                        climb_phase=node.climb_phase,
                        stair_active=node.stair_active,
                        prev_action=action,
                        x=child_info["x_pos"],
                        y=child_info["y_pos"],
                        history=node.history + ((action.value, edge),),
                    )
                )
                if child_info.get("flag_get"):
                    winner = candidates[-1]
                    break
            if winner:
                break
        if winner:
            break
        if not candidates:
            break
        # Dedupe by coarse position, keep the best x per bucket, then cap the beam.
        best_per_bucket: dict[tuple[int, int, bool], Node] = {}
        for cand in candidates:
            bucket = (cand.x // 8, cand.y // 16, cand.info.get("player_state") == 8)
            if bucket not in best_per_bucket or cand.x > best_per_bucket[bucket].x:
                best_per_bucket[bucket] = cand
        beam = sorted(best_per_bucket.values(), key=lambda n: n.x, reverse=True)[:BEAM_WIDTH]
        if depth % 10 == 0:
            print(f"depth={depth} beam_best_x={beam[0].x} beam_size={len(beam)}")

    if winner:
        print(f"WIN: reached flag at x={winner.x} in {len(winner.history)} windows")
        macro = {
            "world": 1,
            "stage": 1,
            "start_x": root.x,
            "steps": [{"action": a, "edge": e} for a, e in winner.history],
        }
        OUT_PATH.write_text(json.dumps(macro, indent=1))
        print(f"macro written to {OUT_PATH}")
    else:
        print(f"no path found; beam died at depth {depth}, best x={beam[0].x if beam else '-'}")
    env.close()


if __name__ == "__main__":
    main()
