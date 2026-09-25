# TypeSafe Mario (Laya Edition)

An experimental controller that lets the [Laya](https://github.com/NandhaKishorM/laya)
decision model directly choose NES controller inputs for the original Super Mario Bros.

The model does **not** receive screenshots. The harness translates emulator telemetry
and RAM into compact, object-centric JSON containing Mario's motion, jump trajectory,
upcoming enemies, terrain, measured response delay, recent-control results, and episode
progress. Laya chooses one of the legal controller actions, the emulator advances several
frames, and the loop repeats. The raw local tile grid remains available in debug logs
and the UI, but is not duplicated in the model input.

Laya runs locally: the checkpoint downloads from Hugging Face on first use and every
decision is a single forward pass — no API key and no network calls at play time.

## Architecture

```text
NES emulator -> telemetry/RAM parser -> structured JSON -> Laya Choice -> controller input
```

The initial action set is intentionally small:

- `noop`
- `right`
- `right_jump`
- `right_run`
- `right_run_jump`
- `jump`
- `left`

## Requirements

- Python 3.13 or newer
- The `laya` package (installed automatically; the checkpoint downloads on first use)
- A legal local setup for Super Mario Bros.

This repository contains no Nintendo ROM or other copyrighted game data. You are
responsible for ensuring that your emulator and game files are obtained and used
lawfully.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[mario,dev]"
```

Inspect the exact JSON and text that will be sent to Laya without launching the
game or running the model:

```bash
.venv/bin/typesafe-mario state-demo
```

Run World 1-1 with Laya making a decision every eight emulator steps. The default
display is a single recordable window with the live game and model telemetry:

```bash
.venv/bin/typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8
```

The dashboard shows the selected action, full Choice probability distribution,
confidence, Laya latency, jump probability, danger score, reward, and parsed game
state. Press `R` or click **Restart** for a fresh episode; the dashboard remains open
after death or level completion. Press `Esc` or `Q` to quit. Use `--display game` for
only the emulator window or `--display none` for a headless benchmark.

Each decision is written to `artifacts/run-<timestamp>.jsonl`. These records include
latency, action probabilities, confidence, canonical model state, raw debug state, and
game outcome, providing the data for a live overlay or rendered social clip.

## What the parser produces

Laya accepts JSON directly, so there is no need to flatten telemetry into prose.
The model-facing object groups observations by meaning:

- `player`: position, velocity, grounded state, jump phase, and power-up
- `trajectory`: airtime, distance since takeoff, and committed gap crossing
- `hazard`: up to three enemies, projected positions, contact timing, and takeoff deadline
- `terrain`: obstacle/gap geometry, observation reliability, and last grounded preview
- `reaction_timing`: action duration and measured observation-to-action delay
- `recent_control`: chosen action, duration, progress gained, and observed outcome
- `episode`: lives, clock, progress, stalls, death, and level completion

For humans, the fuller debug snapshot can still be rendered as compact text:

```text
Goal: Reach the flag in World 1-1 without dying.
Mario: x=172 y=79, moving right, airborne=False, status=small
Progress: 172 (best 172), time=387, lives=2
Nearby enemies: goomba 42px ahead
Local grid (# solid, . empty, E enemy, M Mario):
...........
...........
...........
..M..E.....
###########
```

The structured object is canonical; the text view is only for debugging and UI.

## Laya judgments

Each request evaluates three independent judgments over the same state:

- a `choice` selects the next controller macro;
- a `noul` estimates whether a forward jump is useful now;
- a `score` measures immediate danger for the live visualization.

Exact timing arithmetic stays in code. For example, the parser combines measured
response age, enemy motion, action cadence, and jump-clearance time into a typed
`jump_must_start_this_decision` fact. Laya interprets those facts and owns most
of the controller choice, with a few measured exceptions scripted in `policy.py`:
a wall-climb maneuver for 4-5 tile walls, a panic jump when enemy contact is
inside the reaction horizon, scripted pit takeoffs, and a position-triggered
capped-hop chain for the two fixed 1-1 staircase sections.

## Fine-tuning and the finishing macro

`scripts/finetune_noul.py` re-trains the noul head on rule-derived labels from
run logs (`build`, `train`, `eval` subcommands; encoder hidden states are cached
so head-only training takes minutes on CPU/MPS). Run the result with
`--laya-checkpoint artifacts/laya-mario-noul`.

`scripts/beam_finish.py` drives the policy to the final gauntlet (x≈2080), then
beam-searches decision windows against the live emulator (nes_py savestates) for
a sequence that reaches the flagpole. The verified macro is written to
`artifacts/finish-macro-1-1.json` and replayed with
`--finish-macro artifacts/finish-macro-1-1.json`. With the fine-tuned checkpoint
and the macro, World 1-1 completes (`flag_get=true` at x=3161).

## Development

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```
