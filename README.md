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

The World 1-1 clear uses two artifacts on top of the stock Laya checkpoint:
a **fine-tuned noul head** (`artifacts/laya-mario-noul`) and a **search-verified
finishing macro** (`artifacts/finish-macro-1-1.json`). Neither ships with this
repository; both are reproducible locally with the commands below.

### Why the noul head is fine-tuned

Every decision asks Laya three questions; the `choice` head picks the controller
macro, but actually committing to a jump is gated by the calibrated probability
of the `noul` head ("should Mario start or keep a forward jump now", commit at
p ≥ 0.85). The stock checkpoint's noul calibration is not good enough for that
gate: on logged decisive states it fires one window too early at enemy packs,
drifts to 0.77-0.84 at pit takeoffs (below the commit threshold, measured twice
at the 1-1 pit x≈1126), and wobbles mid-pack. The fine-tune exists to fix that
one head.

### What is trained

`scripts/finetune_noul.py` trains **only the noul head** (encoder detached) on
rule-derived labels from run logs — the labels are not imitation of logged
actions, they are the verified mechanics:

- grounded → jump when the parser's takeoff deadline fires, when a trusted gap
  or wall is within 3 tiles, or when the nearest enemy is within 56 px
  (measured: single goombas are cleared by a takeoff at 40-56 px; the koopa
  jump at 109 px bonked the ? block; the quartet jump at 91 px landed mid-pack);
- airborne → keep holding while crossing a committed gap, and while rising/at
  apex with an enemy within 64 px (holding through the pack turns landings into
  high stomp-bounces; releasing flattens the arc into the gaps between goombas);
- everything else → don't jump.

Training details that matter:

- soft labels 0.99/0.01 instead of hard 0/1 — hard targets destroy calibration
  on a small dataset, and YES states must sit decisively above the 0.85 commit
  threshold after temperature scaling;
- the 421M-parameter encoder runs exactly once per state and its hidden states
  are cached, so head-only training takes minutes on CPU/MPS;
- the noul temperature is re-fit on held-out data after training (the Agent
  divides logits by it at inference; skipping the refit shifts every threshold);
- validation is split by run, not by row, so the same run cannot leak into both
  sides. The current dataset: 4490 train / 177 val states (2751 / 128 yes)
  drawn from 48 logged runs.

Reproduce:

```bash
.venv/bin/python scripts/finetune_noul.py build   # run logs -> rule-labeled dataset
.venv/bin/python scripts/finetune_noul.py train   # dataset -> artifacts/laya-mario-noul
.venv/bin/python scripts/finetune_noul.py eval    # A/B base vs fine-tuned on key states
```

`--multilingual` fine-tunes the mmBERT-base checkpoint instead (2x faster
inference, `artifacts/laya-mario-noul-multilingual`); it trains but has not
cleared World 1-1 — the verified clear uses the english fine-tune.

### The finishing macro

`scripts/beam_finish.py` drives the policy to the final gauntlet (x≈2080), then
beam-searches decision windows against the live emulator (nes_py savestates) for
a sequence that reaches the flagpole. The verified macro is written to
`artifacts/finish-macro-1-1.json` and replayed with
`--finish-macro artifacts/finish-macro-1-1.json`. With the fine-tuned checkpoint
and the macro, World 1-1 completes (`flag_get=true` at x=3161); the recording in
`docs/dashboard-1-1-clear.mp4` shows one such run, decision-for-decision
identical to the headless clear.

## Development

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```
