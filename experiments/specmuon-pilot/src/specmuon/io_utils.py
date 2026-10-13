"""Read-only loaders for frozen external checkpoints.

Nothing in this module writes to the source projects. Integrity of m1 checkpoints is
verified by recomputing the `mte_state_v1` payload sha (hash logic vendored verbatim
from muon_transition_experiment/src/muon_transition/checkpoint.py, file sha256
18d48623b5b01d0be817005855f2a421d24b07b1741eaafd07d669cd5cfc393c).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

LAB_ROOT = Path("/home/b205/projects/training_dynamics_lab")
M1_DIR = LAB_ROOT / "muon_transition_experiment" / "outputs" / "m1"
ROUTING_REPORT = LAB_ROOT / "muon_transition_experiment" / "outputs" / "m0" / "routing_report.json"
V07_DIR = LAB_ROOT / "legacy_fluctuation_response" / "outputs" / "e3_confirmatory" / "v07"


def _payload_sha(obj: dict) -> str:
    """Verbatim logic of mte_state_v1._payload_sha (see module docstring)."""
    h = hashlib.sha256()
    h.update(json.dumps(obj.get("cursor", {}), sort_keys=True, default=str).encode())
    h.update(json.dumps(obj.get("meta", {}), sort_keys=True, default=str).encode())
    for opt_state in obj["optimizer_states"]:
        h.update(json.dumps(opt_state["param_groups"], sort_keys=True, default=str).encode())
        for state in opt_state["state"].values():
            h.update(json.dumps({k: v for k, v in state.items() if not torch.is_tensor(v)},
                                sort_keys=True, default=str).encode())
            for k in sorted(state.keys()):
                if torch.is_tensor(state[k]):
                    h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    for k in sorted(obj["model_state"]):
        t = obj["model_state"][k]
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def muon_param_names() -> list[dict]:
    """The 56 Muon-routed matrices, in optimizer order, from the frozen M0 artifact."""
    rep = json.load(open(ROUTING_REPORT))
    return [e for e in rep["routing_table"] if e["route"] == "muon"]


def load_m1_muon_buffers(run_dir: Path) -> dict:
    """Load + integrity-verify one m1 run's final_state.pt; return momentum buffers.

    Returns {"names": [...], "buffers": {name: fp32 tensor}, "param_groups": [...],
    "cursor": {...}, "file_sha256": ..., "payload_sha256_ok": True}.
    """
    path = Path(run_dir) / "final_state.pt"
    sha = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "mte_state_v1":
        raise RuntimeError(f"unexpected format in {path}")
    stored = payload.pop("payload_sha256")
    ok = stored == _payload_sha(payload)
    if not ok:
        raise RuntimeError(f"payload sha mismatch for {path}")
    opt = payload["optimizer_states"][0]
    if "momentum_buffer" not in next(iter(opt["state"].values())):
        raise RuntimeError("no Muon momentum buffers in optimizer state")
    names = muon_param_names()
    idxs = opt["param_groups"][0]["params"]
    if len(idxs) != len(names):
        raise RuntimeError(f"param count mismatch: {len(idxs)} vs {len(names)}")
    buffers = {}
    for idx, meta in zip(idxs, names):
        buf = opt["state"][idx]["momentum_buffer"]
        if tuple(buf.shape) != tuple(meta["shape"]):
            raise RuntimeError(
                f"shape mismatch for {meta['name']}: {tuple(buf.shape)} vs {meta['shape']}"
            )
        buffers[meta["name"]] = buf.float()
    return {
        "names": [m["name"] for m in names],
        "buffers": buffers,
        "param_groups": opt["param_groups"],
        "cursor": payload["cursor"],
        "file_sha256": sha,
        "payload_sha256_ok": ok,
    }


def v07_param_names(model_state: dict) -> list[str]:
    """model.parameters() order == state_dict key order minus the tied lm_head.weight."""
    return [k for k in model_state.keys() if k != "lm_head.weight"]


def load_v07_state(state_path: Path) -> dict:
    """Load one sealed v07 state; reconstruct A = m_hat / (sqrt(v_hat) + eps)."""
    st = torch.load(state_path, map_location="cpu", weights_only=False)
    pg = st["optimizer"]["param_groups"][0]
    beta1, beta2 = pg["betas"]
    eps = float(pg["eps"])
    names = v07_param_names(st["model"])
    idxs = pg["params"]
    out = {"names": names, "betas": (beta1, beta2), "eps": eps, "A": {}, "raw": {}}
    for idx, name in zip(idxs, names):
        slot = st["optimizer"]["state"][idx]
        t = int(slot["step"].item())
        m, v = slot["exp_avg"].float(), slot["exp_avg_sq"].float()
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        out["A"][name] = m_hat / (v_hat.sqrt() + eps)
        out["raw"][name] = {"m": m, "v": v, "step": t}
    out["step"] = st["metadata"]["state_optimizer_step"]
    out["seed"] = st["metadata"]["train_seed"]
    return out
