#!/usr/bin/env python3
"""Post-hoc screen-split evaluation of layer-1 endpoints (secondary observation only;
selection uses development per frozen_v1.json). Read-only wrt frozen runner; loads saved
endpoints and evaluates the frozen screen 256 docs at T=2048. Identity = endpoint sha."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "raw"
V16 = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
sys.path.insert(0, str(V16))
sys.path.insert(0, str(HERE))
from pilot_core import model_and_optimizer  # noqa: E402
from src.dv_core import eval_docs, token_weighted  # noqa: E402

cfg = json.load(open(HERE / "configs" / "frozen_v1.json", encoding="utf-8"))
screen = json.load(open(V16 / "campaign_v16" / "eval_sets.json", encoding="utf-8"))["screen"]
binding = json.load(open(V16 / "bindings.local.json", encoding="utf-8"))
pcfg = json.load(open(V16 / "config" / "local_v1_lowmem.json", encoding="utf-8"))
model, _, _, _ = model_and_optimizer(pcfg, binding, 0.001, "cuda:0")
out = {}
for arm in cfg["layer1"]["arms"]:
    label = arm["id"]
    ep = RAW / "checkpoints" / f"{label}_endpoint.pt"
    payload = torch.load(ep, map_location="cpu", weights_only=False)
    assert payload["final_step"] == arm["T"]
    model.load_state_dict(payload["model"])
    ident = {"state_id": f"endpoint:{label}",
             "loaded_checkpoint_sha": hashlib.sha256(ep.read_bytes()).hexdigest(),
             "global_step": payload["final_step"],
             "param_fingerprint": payload["param_fingerprint"]}
    rows = eval_docs(model, screen, "screen", "l1screen", label, arm["T"], ident, RAW)
    out[label] = token_weighted(rows)
    print(label, "screen_nll", out[label])
json.dump(out, open(RAW / "layer1_screen_secondary.json", "w"), indent=1)
