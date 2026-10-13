"""V16 campaign core: ledger, resources, data, direction training, evaluation.

Parent trees (V1-V15, artifacts, results) are strictly read-only. Everything new lives
under campaign_v16/. Single-writer ledger; attempts counted before execution."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
V12 = ROOT / "replay_audit_v12"
V14 = ROOT / "mechanism_screen_v14"
V15 = ROOT / "direction_horizon_v15"
CFG = json.loads((HERE / "campaign_config.json").read_text(encoding="utf-8"))
PCFG = json.loads((ROOT / "config" / "local_v1_lowmem.json").read_text(encoding="utf-8"))
TARGET = CFG["target"]
HB = CFG["hard_budget"]
SOURCE_FILES = ("core.py", "run.py", "test_v16.py", "campaign_config.json", "RUN_CAMPAIGN.ps1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    for attempt in range(30):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.05 * (attempt + 1))


def append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def sha(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        left = limit
        while left is None or left > 0:
            block = stream.read(min(left, 1048576) if left is not None else 1048576)
            if not block:
                break
            digest.update(block)
            if left is not None:
                left -= len(block)
    return digest.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_hash():
    return digest({name: sha(HERE / name) for name in SOURCE_FILES if (HERE / name).is_file()})


# ------------------------------------------------------------------ ledger

LEDGER_PATH = HERE / "budget_ledger.json"
_LEDGER_LOCK = HERE / "ledger.lock"


def _fresh_ledger(defaults):
    if LEDGER_PATH.exists():
        return read(LEDGER_PATH)
    value = {"manifest_sha256": None, "wall_started_at": time.time(), **defaults}
    write(LEDGER_PATH, value)
    return value


def ledger():
    value = _fresh_ledger({})
    if value.get("manifest_sha256") is None:
        return value
    expected = sha(HERE / "run_manifest.json") if (HERE / "run_manifest.json").exists() else None
    if expected and value["manifest_sha256"] != expected:
        raise RuntimeError("LEDGER_MANIFEST_MISMATCH")
    return value


@contextlib.contextmanager
def ledger_txn():
    """Single-writer: cross-process lock + fresh read-modify-write."""
    for attempt in range(60):
        try:
            fd = os.open(_LEDGER_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - _LEDGER_LOCK.stat().st_mtime > 120:
                    _LEDGER_LOCK.unlink(missing_ok=True)  # stale lock from a dead process
                    continue
            except OSError:
                pass
            if attempt == 59:
                raise RuntimeError("LEDGER_LOCK_STUCK")
            time.sleep(0.5)
    try:
        value = _fresh_ledger({})
        yield value
        write(LEDGER_PATH, value)
    finally:
        _LEDGER_LOCK.unlink(missing_ok=True)


def count_forward(tokens, phase):
    with ledger_txn() as v:
        if v["token_forward_equivalents"] + tokens > HB["max_token_forward_equivalents"]:
            raise RuntimeError("STOP_BUDGET:TOKEN_FORWARDS")
        v["token_forward_equivalents"] += int(tokens)
        v["forward_calls"] = v.get("forward_calls", 0) + 1
        v["last_phase"] = phase


def bump(**deltas):
    with ledger_txn() as v:
        for key, delta in deltas.items():
            v[key] = v.get(key, 0) + delta


def begin_optimizer(phase):
    with ledger_txn() as v:
        if v["optimizer_step_attempts"] >= HB["max_optimizer_step_attempts"]:
            raise RuntimeError("STOP_BUDGET:OPTIMIZER_ATTEMPTS")
        v["optimizer_step_attempts"] += 1
        v["last_phase"] = phase


def wall_hours():
    return (time.time() - ledger()["wall_started_at"]) / 3600.0


def device_task_hours():
    return ledger().get("device_task_seconds", 0.0) / 3600.0


def budget_exhausted():
    return (wall_hours() > HB["wall_clock_hours"] or
            device_task_hours() > HB["device_task_hours"])


# ------------------------------------------------------------------ resources

VRAM = [None]
_LAST_LOG = [0.0]


def _gpu_probes():
    temp, throttle = None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_throttle_reasons.active",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        temp = int(parts[0])
        throttle = parts[1] if len(parts) > 1 else None
    except Exception:
        pass
    battery = getattr(psutil, "sensors", None)
    on_battery = None
    if battery and battery().get("battery") is not None:
        bat = psutil.sensors_battery()
        on_battery = bool(bat and not bat.power_plugged)
    return temp, throttle, on_battery


def resource(phase, admission=False, force=False):
    ram = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    initialized = torch.cuda.is_initialized()
    reserved = torch.cuda.memory_reserved(0) if initialized else None
    free = torch.cuda.mem_get_info(0)[0] if initialized else None
    minimum = HB["ram_start_min_gib" if admission else "ram_hard_gib"] * 2 ** 30
    reason = ("RAM_BELOW_REQUIRED" if ram < minimum else
              "RSS_ABOVE_LIMIT" if rss > HB["rss_limit_gib"] * 2 ** 30 else
              "VRAM_ABOVE_LIMIT" if VRAM[0] and reserved is not None and reserved > VRAM[0] else None)
    temp, throttle, on_battery = _gpu_probes()
    if reason is None and temp is not None and temp >= HB["gpu_temp_pause_c"]:
        reason = f"GPU_TEMP_{temp}C"
    # Only thermal/hardware slowdown bits block: 0x8 HwSlowdown, 0x20 SwThermalSlowdown,
    # 0x40 HwThermalSlowdown, 0x80 HwPowerBrakeSlowdown. GpuIdle(0x1)/SwPowerCap(0x4) etc.
    # are normal idle/power states, not thermal throttling.
    if reason is None and throttle:
        try:
            bits = int(str(throttle), 0) if not throttle.startswith("0x") else int(throttle, 16)
        except ValueError:
            bits = 0
        blocking = bits & 0xE8
        if blocking:
            reason = f"GPU_THROTTLE_0x{blocking:X}"
    if reason is None and on_battery:
        reason = "ON_BATTERY"
    status = "BLOCKED" if reason else ("WARNING" if not admission and ram < HB["ram_warning_gib"] * 2 ** 30 else "READY")
    now = time.time()
    if force or now - _LAST_LOG[0] >= CFG["storage"]["resource_log_interval_seconds"] or status != "READY":
        l = ledger()
        append(HERE / "logs" / "resources.jsonl",
               {"time": now, "phase": phase, "status": status, "reason": reason,
                "ram_available_bytes": ram, "rss_bytes": rss, "cuda_reserved_bytes": reserved,
                "cuda_free_bytes": free, "gpu_temp_c": temp, "throttle": throttle,
                "on_battery": on_battery,
                "optimizer_step_attempts": l.get("optimizer_step_attempts", 0),
                "token_forward_equivalents": l.get("token_forward_equivalents", 0),
                "device_task_seconds": l.get("device_task_seconds", 0.0)})
        _LAST_LOG[0] = now
    if reason:
        raise RuntimeError("RESOURCE_BLOCKED:" + reason)
    return status


class PauseBudget:
    def __init__(self):
        self.total = 0.0

    def wait_for_recovery(self, hard_deadline=None):
        """Returns True (recovered), False (2h episode cap exhausted) or 'deadline'."""
        deadline_total = HB["pause_total_limit_hours"] * 3600
        good = 0
        while self.total < deadline_total:
            if hard_deadline is not None and time.time() >= hard_deadline:
                return "deadline"
            time.sleep(HB["pause_recheck_seconds"])
            self.total += HB["pause_recheck_seconds"]
            bump(pause_seconds=HB["pause_recheck_seconds"])
            ram = psutil.virtual_memory().available
            vram_ok = True
            if torch.cuda.is_initialized():
                free = torch.cuda.mem_get_info(0)[0]
                vram_ok = free >= HB["vram_admission_min_free_gib"] * 2 ** 30
            temp, _, on_battery = _gpu_probes()
            if ram >= 1.6 * 2 ** 30 and vram_ok and (temp is None or temp < HB["gpu_temp_pause_c"]) and not on_battery:
                good += 1
                if good >= 2:
                    return True
            else:
                good = 0
        return False


def disk_free_gib():
    try:
        return psutil.disk_usage(str(HERE)).free / 2 ** 30
    except OSError:
        return None


def dir_size_gib(path=HERE):
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 2 ** 30


# ------------------------------------------------------------------ state files

def state():
    return read(HERE / "STATE.json") if (HERE / "STATE.json").exists() else {}


def save_state(**fields):
    value = state()
    value.update(fields)
    write(HERE / "STATE.json", value)


def queue_rows():
    if not (HERE / "QUEUE.jsonl").exists():
        return []
    return [json.loads(line) for line in (HERE / "QUEUE.jsonl").open(encoding="utf-8")]


def queue_add(task):
    append(HERE / "QUEUE.jsonl", {"id": task["id"], "status": "pending", **task})


def queue_set_status(task_id, status, note=None):
    rows = queue_rows()
    out = []
    for row in rows:
        if row["id"] == task_id:
            row["status"] = status
            if note:
                row["note"] = note
        out.append(row)
    with (HERE / "QUEUE.jsonl").open("w", encoding="utf-8") as stream:
        for row in out:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def decision(condition, evidence, action):
    append(HERE / "DECISIONS.md", f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')}\n- 条件: {condition}\n"
                                  f"- 证据: {evidence}\n- 行动: {action}\n")


def log_deletion(path, reason, sha256=None):
    append(HERE / "deletions.jsonl", {"time": time.time(), "path": str(path),
                                       "sha256": sha256 or (sha(path) if Path(path).exists() else None),
                                       "reason": reason})


# ------------------------------------------------------------------ model / data

def model_and_optimizer(device, *, peak_lr=0.003, target_beta1=None, all_beta1=None, wd_matrix=None):
    """Parent model/optimizer; optional per-param beta1 or matrix-WD variants built as
    one param-group per parameter so group betas/wd are exact."""
    from pilot_core import model_and_optimizer as parent_mao
    binding = read(ROOT / "bindings.local.json")
    model, opt, aliases, count = parent_mao(PCFG, binding, peak_lr, device)
    if target_beta1 is None and all_beta1 is None and wd_matrix is None:
        return model, opt, aliases, count
    base_beta1 = all_beta1 if all_beta1 is not None else PCFG["baseline"]["betas"][0]
    beta2 = PCFG["baseline"]["betas"][1]
    target = dict(model.named_parameters())[TARGET]
    matrix_wd = wd_matrix if wd_matrix is not None else PCFG["baseline"]["matrix_weight_decay"]
    groups = []
    for name, p in model.named_parameters():
        wd = matrix_wd if p.ndim >= 2 else 0.0
        b1 = target_beta1 if (p is target and target_beta1 is not None) else base_beta1
        groups.append({"params": [p], "lr": peak_lr, "weight_decay": wd, "betas": (b1, beta2)})
    opt = torch.optim.AdamW(groups, lr=peak_lr, betas=(base_beta1, beta2),
                            eps=PCFG["baseline"]["eps"], amsgrad=False, maximize=False,
                            foreach=False, fused=False)
    return model, opt, aliases, count


def schedule_eta(peak, k, T, warmup, ratio):
    if k <= warmup:
        return peak * k / warmup
    return peak * (ratio + (1 - ratio) * (1 + math.cos(math.pi * (k - warmup) / (T - warmup))) / 2)


def parent_schedule_eta(peak, k):
    from pilot_core import schedule_lr
    return schedule_lr(peak, k, PCFG)


def n_a(x, a_norm):
    xn = float(torch.linalg.vector_norm(x.double()))
    if xn == 0.0 or a_norm == 0.0:
        return torch.zeros_like(x), True, xn
    return (x.double() * (a_norm / xn)).to(x.dtype), False, xn


ANCHOR_KEYS = [CFG["target"], CFG["adjacent_ffn"], CFG["attention_output"]]


def apply_step(model, opt, family, a, b, phase, eta, target_name=TARGET, q_override=None,
               anchor_recorder=None, anchor_names=()):
    """Native opt.step() + verified displacement replacement. Returns meta with q_t.
    anchor_names: when non-empty, m_plus for those matrices is captured into meta for the
    anchor recorder (cheap; only requested at anchor steps by the caller)."""
    named = dict(model.named_parameters())
    beta1, beta2 = PCFG["baseline"]["betas"]
    eps = PCFG["baseline"]["eps"]
    per_beta1 = {}
    for group in opt.param_groups:
        if len(group["params"]) == 1 and group["betas"][0] != beta1:
            per_beta1[id(group["params"][0])] = group["betas"][0]
    A_map, meta = {}, {"family": family, "_m_plus": {}}
    for name, param in named.items():
        if param.grad is None:
            A_map[name] = None
            continue
        b1 = per_beta1.get(id(param), beta1)
        st = opt.state[param]
        if "step" in st:
            step_plus = int(st["step"].item() if torch.is_tensor(st["step"]) else st["step"]) + 1
            m_prev, v_prev = st["exp_avg"], st["exp_avg_sq"]
        else:
            step_plus, m_prev, v_prev = 1, torch.zeros_like(param), torch.zeros_like(param)
        m_plus = m_prev * b1 + param.grad * (1 - b1)
        v_plus = v_prev * beta2 + param.grad.square() * (1 - beta2)
        A_map[name] = (m_plus / (1 - b1 ** step_plus)) / ((v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
        if name == target_name:
            meta["moments_step_plus"] = step_plus
        if anchor_names and name in anchor_names:
            meta["_m_plus"][name] = m_plus.detach().cpu().clone()
    target_A = A_map[target_name]
    meta["A_norm"] = float(torch.linalg.vector_norm(target_A.double()))
    if family in ("F", "P"):
        g_clip = named[target_name].grad.detach()
        if family == "P":
            b1 = per_beta1.get(id(named[target_name]), beta1)
            st = opt.state[named[target_name]]
            v_plus = (st["exp_avg_sq"] * beta2 + g_clip.square() * (1 - beta2)) if "step" in st \
                else g_clip.square() * (1 - beta2)
            step_plus = meta["moments_step_plus"]
            x = g_clip / ((v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
        else:
            x = g_clip
        D, zero_fb, xn = n_a(x, meta["A_norm"])
        meta.update(zero_norm_fallback=zero_fb, input_norm=xn)
    else:
        D = target_A
        meta.update(zero_norm_fallback=False, input_norm=meta["A_norm"])
    if q_override is not None:
        dn = float(torch.linalg.vector_norm(D.double()))
        # q_override is the FULL target displacement norm ||eta*a*D||; rescale D so the
        # final displacement equals it exactly (no repeated eta/a multiplication).
        desired_D = (q_override / (eta * a)) if (a != 0 and eta != 0) else 0.0
        if dn > 0 and desired_D > 0:
            D = (D.double() * (desired_D / dn)).to(D.dtype)
        elif desired_D == 0:
            D = torch.zeros_like(D)
        meta["q_override"] = q_override
    if not torch.isfinite(D).all() or not torch.isfinite(target_A).all():
        raise RuntimeError("NON_FINITE_DIRECTION")
    begin_optimizer(phase)
    started = time.perf_counter()
    opt.step()
    bump(optimizer_step_completions=1, device_task_seconds=time.perf_counter() - started)
    torch.cuda.synchronize()
    with torch.no_grad():
        for name, param in named.items():
            A_p = A_map[name]
            if A_p is None:
                continue
            if name == target_name:
                if not (family == "S" and a == 1.0 and q_override is None):
                    param.add_(eta * (target_A - a * D))
            elif b != 1.0:
                param.add_(eta * (1.0 - b) * A_p)
    torch.cuda.synchronize()
    meta["D_norm"] = float(torch.linalg.vector_norm(D.double()))
    meta["q_t"] = eta * a * meta["D_norm"]
    if anchor_recorder is not None:
        anchor_recorder(named, A_map, target_A, meta)
    return meta


# ------------------------------------------------------------------ evaluation

def param_fingerprint(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def eval_docs(model, docs, split, phase, label, horizon, identity):
    """identity: {'state_id', 'loaded_checkpoint_sha', 'global_step', 'param_fingerprint'}"""
    from pilot_core import loss_sum_count
    if identity.get("loaded_checkpoint_sha") is None and not str(identity.get("state_id", "")).startswith("scratch_"):
        raise RuntimeError("EVAL_WITHOUT_STATE_ORIGIN")
    path = HERE / "raw" / "eval" / f"{label}__H{horizon}__{split}.jsonl"
    old = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    if [x["doc_id"] for x in old] != [x["doc_id"] for x in docs[: len(old)]]:
        raise RuntimeError("PARTIAL_EVAL_IDENTITY_MISMATCH")
    if len(old) == len(docs):
        return old
    model.eval()
    started = time.perf_counter()
    for doc in docs[len(old):]:
        resource(f"{phase}_eval")
        ids = torch.tensor(doc["ids"], device="cuda:0", dtype=torch.long)[None, :]
        count_forward(doc["token_count"], phase)
        with torch.no_grad():
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
        if count != doc["token_count"]:
            raise RuntimeError("MASK_OR_SHIFT_CHANGED")
        append(path, {"branch": label, "horizon": horizon, "split": split, "doc_id": doc["doc_id"],
                      "loss_sum": float(loss), "token_count": int(count), **identity})
    torch.cuda.synchronize()
    bump(eval_seconds=time.perf_counter() - started, device_task_seconds=0.0)
    return old


def eval_rows(label, horizon, split):
    return [json.loads(line) for line in (HERE / "raw" / "eval" / f"{label}__H{horizon}__{split}.jsonl").open(encoding="utf-8")]


def token_weighted(rows):
    return sum(r["loss_sum"] for r in rows) / sum(r["token_count"] for r in rows)


def paired_bootstrap(rows_a, rows_b, seed=None, reps=None):
    stats = CFG["stats"]
    if [r["doc_id"] for r in rows_a] != [r["doc_id"] for r in rows_b]:
        raise RuntimeError("PAIRING_MISMATCH")
    la = np.array([r["loss_sum"] for r in rows_a]); lb = np.array([r["loss_sum"] for r in rows_b])
    ta = np.array([r["token_count"] for r in rows_a]); tb = np.array([r["token_count"] for r in rows_b])
    point = la.sum() / ta.sum() - lb.sum() / tb.sum()
    rng = np.random.default_rng(seed if seed is not None else stats["bootstrap_seed"])
    reps = reps if reps is not None else stats["bootstrap_repetitions"]
    draws = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(rows_a), len(rows_a))
        draws[i] = la[idx].sum() / ta[idx].sum() - lb[idx].sum() / tb[idx].sum()
    return {"point": point,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "ci97p5": [float(np.quantile(draws, 0.0125)), float(np.quantile(draws, 0.9875))]}
