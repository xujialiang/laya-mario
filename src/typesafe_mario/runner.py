from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from .dashboard import DashboardCommand, LiveDashboard
from .policy import Decision, Policy
from .state import MarioStateParser


def _unwrap_ram(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return None


def create_mario_env(env_id: str, render_mode: str = "human") -> Any:
    try:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            'Mario dependencies are missing. Install with: pip install -e ".[mario]"'
        ) from exc

    env = gym.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def _record_decision(
    log: Any,
    *,
    decision_index: int,
    snapshot: Any,
    decision: Decision,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    record = {
        "decision": decision_index,
        "state": snapshot.to_state(),
        "debug_state": snapshot.to_debug_state(),
        "state_text": snapshot.to_text(),
        "action": decision.action.value,
        "confidence": decision.confidence,
        "probabilities": dict(decision.probabilities),
        "jump_needed_probability": decision.jump_needed_probability,
        "danger_score": decision.danger_score,
        "latency_ms": decision.latency_ms,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_realtime_dashboard(
    *,
    env: Any,
    dashboard: LiveDashboard,
    policy: Policy,
    parser: MarioStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
) -> None:
    """Realtime display loop built on the headless execution primitives.

    Decisions are synchronous and each action window is executed by
    ``step_decision_window`` — the same code path as ``--display none`` —
    with a per-frame callback that draws the dashboard between emulator
    frames. The picture simply freezes while Laya is thinking.
    """
    actions = tuple(Action)
    episode_reward = 0.0
    previous_action: Action | None = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    frames_since_parse = 0
    screenshot_saved = False
    decision_index = 0
    ended = False

    while True:
        snapshot = parser.parse(
            info,
            _unwrap_ram(env),
            previous_action=previous_action.value if previous_action else None,
            previous_reward=previous_reward,
            previous_latency_ms=previous_latency_ms,
            previous_response_delay_frames=0,
            elapsed_frames=frames_since_parse,
        )
        if not ended and (snapshot.dead or snapshot.clear):
            ended = True

        if ended or decision_index >= max_decisions:
            command = dashboard.draw(
                frame,
                snapshot,
                None,
                decision_index=decision_index,
                episode_reward=episode_reward,
                waiting=False,
                run_ended=True,
            )
            if command == DashboardCommand.QUIT:
                break
            if command == DashboardCommand.RESTART:
                frame, info = env.reset()
                parser.reset()
                previous_action = None
                previous_reward = 0.0
                previous_latency_ms = 0.0
                frames_since_parse = 0
                episode_reward = 0.0
                decision_index = 0
                ended = False
                print("--- Restarted ---")
            continue

        decision = policy.choose(snapshot, actions)
        display: dict[str, DashboardCommand] = {"command": DashboardCommand.CONTINUE}

        def _on_frame(rendered: Any) -> None:
            display["command"] = dashboard.draw(
                rendered,
                snapshot,
                decision,
                decision_index=decision_index,
                episode_reward=episode_reward,
                waiting=False,
                run_ended=False,
            )

        (
            frame,
            info,
            terminated,
            truncated,
            stepped,
            _landed,
            total_reward,
        ) = step_decision_window(
            env,
            info,
            action=decision.action,
            force_edge=decision.force_button_edge,
            previous_action=previous_action,
            snapshot=snapshot,
            frames_per_decision=frames_per_decision,
            on_frame=_on_frame,
        )
        frames_since_parse = stepped

        _record_decision(
            log,
            decision_index=decision_index,
            snapshot=snapshot,
            decision=decision,
            reward=total_reward,
            terminated=terminated,
            truncated=truncated,
        )
        print(
            f"#{decision_index:04d} x={snapshot.x:04d} "
            f"action={decision.action.value:<15} "
            f"confidence={decision.confidence:.2f} "
            f"latency={decision.latency_ms:.0f}ms"
        )
        if screenshot_path is not None and not screenshot_saved:
            dashboard.save(screenshot_path)
            screenshot_saved = True

        episode_reward += total_reward
        previous_action = decision.action
        previous_reward = total_reward
        previous_latency_ms = decision.latency_ms
        decision_index += 1
        if terminated or truncated:
            ended = True
        if display["command"] == DashboardCommand.QUIT:
            break
        if display["command"] == DashboardCommand.RESTART:
            frame, info = env.reset()
            parser.reset()
            previous_action = None
            previous_reward = 0.0
            previous_latency_ms = 0.0
            frames_since_parse = 0
            episode_reward = 0.0
            decision_index = 0
            ended = False
            print("--- Restarted ---")


def step_decision_window(
    env: Any,
    info: dict[str, Any],
    *,
    action: Action,
    force_edge: bool,
    previous_action: Action | None,
    snapshot: Any,
    frames_per_decision: int,
    on_frame: Any = None,
) -> tuple[Any, dict[str, Any], bool, bool, int, bool, float]:
    """Step one decision window; the single source of macro execution semantics.

    Applies the jump button-up edge (forced, or when the previous macro held A and
    Mario is grounded) and the landing break (a window that started falling ends
    at touchdown so the next decision happens on the ground). ``on_frame``, when
    given, is called with each rendered frame so the realtime dashboard can draw
    between emulator steps without altering the window semantics. Returns
    (frame, info, terminated, truncated, stepped, landed_mid_window, reward).
    """
    stepped = 0
    total_reward = 0.0
    terminated = truncated = False
    frame = None

    def _step(input_action: Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_frame, reward, step_terminated, step_truncated, step_info = env.step(
            ACTION_TO_INDEX[input_action]
        )
        if on_frame is not None:
            on_frame(step_frame)
        return step_frame, float(reward), step_terminated, step_truncated, step_info

    if action in JUMP_ACTIONS and (
        force_edge or (previous_action in JUMP_ACTIONS and snapshot.grounded)
    ):
        # A fresh jump needs a button-up edge: Mario landing with A still held
        # from the previous macro cannot re-jump and runs into whatever he meant
        # to clear (measured at the 1-1 koopa).
        frame, reward, terminated, truncated, info = _step(JUMP_RELEASE_ACTION[action])
        total_reward += reward
        stepped += 1
    # Landing into an enemy's face needs an immediate fresh decision: the blind
    # 8-frame window otherwise runs Mario into the goomba before he can jump
    # (measured at the 1-1 pair, x~1503). Only break when a same-level enemy is
    # close, so ordinary landings keep the cadence.
    threat = snapshot.threat_features()
    started_falling = (
        snapshot.jump_phase == "falling"
        and threat["enemy_ahead"]
        and (threat["nearest_enemy_distance_pixels"] or 999) <= 48
    )
    prev_y = info["y_pos"]
    landed = False
    while stepped < frames_per_decision and not (terminated or truncated):
        frame, reward, terminated, truncated, info = _step(action)
        total_reward += reward
        stepped += 1
        if started_falling and info["y_pos"] >= prev_y:
            landed = True
            break  # touchdown: decide the landing jump right now
        prev_y = info["y_pos"]
    return frame, info, terminated, truncated, stepped, landed, total_reward


def run_episode(
    *,
    env_id: str,
    policy: Policy,
    frames_per_decision: int,
    max_decisions: int,
    seed: int,
    artifacts_dir: Path,
    display: str = "dashboard",
    screenshot_path: Path | None = None,
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    render_mode = "human" if display == "game" else "rgb_array"
    env = create_mario_env(env_id, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    parser = MarioStateParser(decision_horizon_frames=frames_per_decision)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    frame, info = env.reset(seed=seed)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            if dashboard:
                _run_realtime_dashboard(
                    env=env,
                    dashboard=dashboard,
                    policy=policy,
                    parser=parser,
                    frame=frame,
                    info=info,
                    log=log,
                    frames_per_decision=frames_per_decision,
                    max_decisions=max_decisions,
                    screenshot_path=screenshot_path,
                )
                return log_path

            previous_action: Action | None = None
            previous_reward = 0.0
            previous_latency_ms = 0.0
            actions = tuple(Action)
            frames_since_parse = 0
            for decision_index in range(max_decisions):
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

                decision = policy.choose(snapshot, actions)
                (
                    frame,
                    info,
                    terminated,
                    truncated,
                    stepped,
                    _landed,
                    total_reward,
                ) = step_decision_window(
                    env,
                    info,
                    action=decision.action,
                    force_edge=decision.force_button_edge,
                    previous_action=previous_action,
                    snapshot=snapshot,
                    frames_per_decision=frames_per_decision,
                )
                frames_since_parse = stepped

                _record_decision(
                    log,
                    decision_index=decision_index,
                    snapshot=snapshot,
                    decision=decision,
                    reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                print(
                    f"#{decision_index:04d} x={snapshot.x:04d} "
                    f"action={decision.action.value:<15} "
                    f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                )

                previous_action = decision.action
                previous_reward = total_reward
                previous_latency_ms = decision.latency_ms
                if terminated or truncated:
                    break
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
