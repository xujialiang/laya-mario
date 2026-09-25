"""Fine-tune the Laya english checkpoint's noul head on Mario run logs.

The pip package ships no training API (RLAgent is just an alias of Agent), so this
script follows the official fine-tuning notebook's approach with the package's own
building blocks (build_sequence / collate_items / build_model). Labels are not
imitation of the logged actions - they are rule-derived from verified mechanics:

- grounded: jump when the parser's takeoff deadline fires, when a trusted gap or
  wall is within 3 tiles, or when the nearest enemy is within 56 px (measured:
  single goombas are cleared by a takeoff at ~40-56 px; the koopa jump at 109 px
  bonked the ? block, and the goomba quartet jump at 91 px landed mid-pack);
- airborne: keep holding over a committed gap crossing, and while any enemy is
  within 64 px (oracle replay: holding through the quartet turns landings into
  high stomp-bounces that clear the pack; releasing flattens the arc into the
  gaps between goombas);
- everything else: don't jump.

Only the head is trained (encoder detached): a few thousand states is far too
little to touch a 421M encoder, and head-only training runs in minutes.

Usage:
    .venv/bin/python scripts/finetune_noul.py build   # logs -> dataset
    .venv/bin/python scripts/finetune_noul.py train   # dataset -> checkpoint
    .venv/bin/python scripts/finetune_noul.py eval    # A/B key logged states
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from typesafe_mario.policy import _JUMP_INSTRUCTIONS, _narrative_state

ARTIFACTS = ROOT / "artifacts"
DATASET_PATH = ARTIFACTS / "noul_dataset.jsonl"
CHECKPOINT_OUT = ARTIFACTS / "laya-mario-noul"

# Set by --multilingual: fine-tune the mmBERT-base checkpoint (2x faster
# inference) from the repo's multilingual/ subfolder instead of the english root.
MULTILINGUAL = False

# Label smoothing: hard 0/1 targets destroy calibration on a tiny dataset, but
# the harness commits jumps at p>=0.85, so YES states must sit decisively above
# it after temperature scaling - 0.95 targets left deadline states at ~0.849.
YES, NO = 0.99, 0.01

# Grounded: commit when the nearest enemy is this close. Measured: single goombas
# are stomped cleanly by a takeoff at 40-56 px; the koopa jump at 109 px bonked
# the overhead block and the quartet jump at 91 px landed mid-pack.
_ENEMY_COMMIT_PIXELS = 56
# Airborne: keep the button held while an enemy is inside the landing zone, so a
# touchdown on a head bounces high instead of skidding into the next goomba.
_ENEMY_HOLD_PIXELS = 64


def noul_target(state: dict) -> float | None:
    """Rule-derived soft label for 'should Mario start/keep a forward jump now'."""
    episode = state["episode"]
    if episode["dead"] or episode["stage_clear"]:
        return None
    player = state["player"]
    hazard = state["hazard"]
    terrain = state["terrain"]
    trajectory = state["trajectory"]

    if not player["grounded"]:
        if trajectory["crossing_known_gap"]:
            return YES
        # Hold only while the button still does something: rising or at the apex
        # with an enemy inside the arc's reach. Holding while falling is
        # physically inert (A only shapes the rise) and actively harmful - it
        # means Mario lands with A held and cannot re-jump (measured: pit arc
        # landed him in front of the goomba pair, button held, no edge, dead).
        if (
            player["jump_phase"] in ("rising", "apex")
            and hazard["enemy_ahead"]
            and (hazard.get("nearest_enemy_distance_pixels") or 999) <= _ENEMY_HOLD_PIXELS
        ):
            return YES
        return NO

    if hazard["jump_must_start_this_decision"]:
        return YES
    if terrain["gap_ahead"] or terrain["obstacle_ahead"]:
        return YES
    if hazard["enemy_ahead"]:
        distance = hazard.get("nearest_enemy_distance_pixels")
        if distance is not None:
            return YES if distance <= _ENEMY_COMMIT_PIXELS else NO
    return NO


def build_dataset() -> None:
    rows: list[dict] = []
    for log_path in sorted(ARTIFACTS.glob("run-*.jsonl")):
        for line in log_path.open():
            record = json.loads(line)
            target = noul_target(record["state"])
            if target is None:
                continue
            rows.append(
                {
                    "run": log_path.name,
                    "decision": record["decision"],
                    "state": _narrative_state(record["state"]),
                    "target": target,
                }
            )

    # Downsample the dominant class (plain flat-ground running) so the head does
    # not just learn "always no": keep every YES and every enemy-adjacent NO, and
    # one in four of the rest.
    kept = [
        row
        for row in rows
        if row["target"] == YES
        or "goomba" in row["state"]
        or "koopa" in row["state"]
        or random.random() < 0.25
    ]
    random.shuffle(kept)

    runs = sorted({row["run"] for row in kept})
    val_runs = set(runs[-2:])  # split by run, not by row, or the same run leaks
    train = [row for row in kept if row["run"] not in val_runs]
    val = [row for row in kept if row["run"] in val_runs]

    with DATASET_PATH.open("w") as out:
        for row in train:
            out.write(json.dumps({**row, "split": "train"}) + "\n")
        for row in val:
            out.write(json.dumps({**row, "split": "val"}) + "\n")

    yes_train = sum(row["target"] == YES for row in train)
    yes_val = sum(row["target"] == YES for row in val)
    print(f"dataset: {len(train)} train ({yes_train} yes), {len(val)} val ({yes_val} yes)")
    print(f"written to {DATASET_PATH}")


def train() -> None:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from laya.common import QTYPES, build_model, build_sequence
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    src_dir = Path(snapshot_download("convaiinnovations/laya"))
    if MULTILINGUAL:
        src_dir = src_dir / "multilingual"
    cfg = json.loads((src_dir / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(src_dir / "tokenizer")

    dataset = [json.loads(line) for line in DATASET_PATH.open()]
    rows = [(row, row["split"]) for row in dataset]

    question = {"t": "noul", "ins": _JUMP_INSTRUCTIONS}

    # Tokenize every state once. max_len follows the base checkpoint (512 english,
    # 1024 multilingual); head_max_len matches the policy's inference budget.
    encoded = []
    for row, split in rows:
        ids, markers = build_sequence(
            tok, row["state"], question, max_len=cfg["max_len"], head_max_len=256
        )
        encoded.append(
            {
                "ids": ids,
                "markers": markers,
                "target": [1.0 - row["target"], row["target"]],
                "label": int(row["target"] > 0.5),
                "split": split,
            }
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    model = build_model(cfg, str(src_dir / "encoder"))
    state = load_file(src_dir / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)

    # The encoder output is identical every epoch under head-only training, so run
    # the 421M encoder exactly once per state and cache the hidden states. Without
    # this, each of the 4x113 training steps re-runs the full encoder - that was
    # the entire training time. Head steps on cached embeddings take milliseconds.
    print("encoding states (one-time encoder pass)...")
    cache = []
    t0 = time.time()
    with torch.no_grad():
        model.eval()
        for start in range(0, len(encoded), 64):
            chunk = encoded[start : start + 64]
            length = max(len(it["ids"]) for it in chunk)
            ids = torch.full((len(chunk), length), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), length), dtype=torch.long)
            for i, it in enumerate(chunk):
                ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
                att[i, : len(it["ids"])] = 1
            hidden = model.encoder(
                input_ids=ids.to(device), attention_mask=att.to(device)
            ).last_hidden_state
            for i, it in enumerate(chunk):
                cache.append(
                    {
                        "h": hidden[i, : len(it["ids"])].cpu().half(),
                        "markers": it["markers"],
                        "target": it["target"],
                        "label": it["label"],
                        "split": it["split"],
                    }
                )
    print(f"encoded {len(cache)} states in {time.time() - t0:.0f}s")

    train_items = [c for c in cache if c["split"] == "train"]
    val_items = [c for c in cache if c["split"] == "val"]

    def head_forward(chunk: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replicate DecisionModel.forward post-encoder on cached hidden states."""
        length = max(c["h"].size(0) for c in chunk)
        width = cache[0]["h"].size(1)
        h = torch.zeros(len(chunk), length, width)
        att = torch.zeros(len(chunk), length, dtype=torch.long)
        mpos = torch.zeros(len(chunk), 2, dtype=torch.long)
        for i, c in enumerate(chunk):
            h[i, : c["h"].size(0)] = c["h"].float()
            att[i, : c["h"].size(0)] = 1
            mpos[i] = torch.tensor(c["markers"])
        h = (
            h.to(device)
            + model.type_emb(torch.full((len(chunk),), QTYPES["noul"], device=device))[:, None, :]
        )
        pad = ~att.bool().to(device)
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)).to(device)
        pooled = torch.gather(h, 1, idx)
        logits = model.scorer(pooled).squeeze(-1).float()
        target = torch.tensor([c["target"] for c in chunk], device=device)
        label = torch.tensor([c["label"] for c in chunk])
        return logits, target, label

    trained = list(model.head.parameters()) + list(model.scorer.parameters())
    optimizer = torch.optim.AdamW(trained, lr=1e-3, weight_decay=0.01)

    model.train()
    epochs, batch_size = 6, 64
    for epoch in range(epochs):
        random.shuffle(train_items)
        total_loss = correct = 0
        for start in range(0, len(train_items), batch_size):
            logits, target, label = head_forward(train_items[start : start + batch_size])
            loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(-1).cpu() == label).sum().item()
        print(
            f"epoch {epoch}: loss={total_loss / len(train_items):.4f} acc={correct / len(train_items):.3f}"
        )

    # Held-out accuracy plus the raw-logit temperature that minimises NLL. The
    # Agent divides logits by this temperature at inference, so it must be refit
    # after head-only training or every downstream threshold shifts.
    model.eval()
    logits, targets, labels = head_forward(val_items)
    logits, targets = logits.cpu(), targets.cpu()
    accuracy = (logits.argmax(-1) == labels).float().mean().item()
    best_t, best_nll = 1.0, float("inf")
    for t in [x / 20 for x in range(10, 81)]:
        nll = -(targets * F.log_softmax(logits / t, dim=-1)).sum(-1).mean().item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    print(f"val: acc={accuracy:.3f} noul_temperature={best_t:.2f} nll={best_nll:.4f}")

    CHECKPOINT_OUT.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
        CHECKPOINT_OUT / "model.safetensors",
    )
    shutil.copytree(src_dir / "tokenizer", CHECKPOINT_OUT / "tokenizer", dirs_exist_ok=True)
    shutil.copytree(src_dir / "encoder", CHECKPOINT_OUT / "encoder", dirs_exist_ok=True)
    cfg_out = dict(cfg)
    cfg_out["temperature"] = [*cfg["temperature"][:2], best_t]
    cfg_out["fine_tuned"] = "mario-noul-head-only"
    (CHECKPOINT_OUT / "rl_agent_config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"checkpoint written to {CHECKPOINT_OUT}")


def eval_states() -> None:
    """A/B the base and fine-tuned checkpoints on the decisive logged states."""
    from laya import Agent

    cases = [
        # (run file, decision, what the verified-correct answer is)
        ("run-20260924T175015Z.jsonl", 90, "no - pack takeoff one window too early"),
        ("run-20260924T175015Z.jsonl", 92, "yes - airborne over the pack, hold"),
        ("run-20260924T175015Z.jsonl", 93, "yes - descending onto a goomba, hold"),
        ("run-20260924T170129Z.jsonl", 82, "no - koopa 109 px out, too early"),
        ("run-20260924T170129Z.jsonl", 85, "yes - koopa at the deadline"),
        ("run-20260924T170902Z.jsonl", 82, "yes - airborne over the pit, hold"),
    ]
    questions = {"jump_needed": {"type": "noul", "instructions": _JUMP_INSTRUCTIONS}}
    base_kwargs = {"subfolder": "multilingual"} if MULTILINGUAL else {}
    for name, path in (("base", "convaiinnovations/laya"), ("finetuned", str(CHECKPOINT_OUT))):
        agent = Agent(path, **(base_kwargs if name == "base" else {}))
        line = []
        for run, decision, note in cases:
            for row in (json.loads(l) for l in (ARTIFACTS / run).open()):
                if row["decision"] == decision:
                    p = agent.predict(_narrative_state(row["state"]), questions)["answers"][
                        "jump_needed"
                    ]["noul"]
                    line.append(f"d{decision}={p:.2f}")
                    break
        print(f"{name:10s}: {'  '.join(line)}")
        unload = getattr(agent, "unload", None)
        if callable(unload):
            unload()
    print(
        "expected:       "
        + "  ".join(f"d{d}={'>=0.85' if 'yes' in n else '<0.85'}" for _, d, n in cases)
    )
    print("(" + "; ".join(f"d{d} {n}" for _, d, n in cases) + ")")


def main() -> None:
    global CHECKPOINT_OUT, MULTILINGUAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "train", "eval"))
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Fine-tune the mmBERT-base multilingual checkpoint (2x faster inference)",
    )
    args = parser.parse_args()
    if args.multilingual:
        MULTILINGUAL = True
        CHECKPOINT_OUT = ARTIFACTS / "laya-mario-noul-multilingual"
    random.seed(0)
    if args.command == "build":
        build_dataset()
    elif args.command == "train":
        train()
    else:
        eval_states()


if __name__ == "__main__":
    main()
