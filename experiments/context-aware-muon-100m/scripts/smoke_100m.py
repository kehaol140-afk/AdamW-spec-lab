"""Limited GPU smoke for context_aware_muon (100M, GPU0 only).

Three optimizer paths x 30 steps x 8192 tokens/step (micro 8 x accum 4, seq 256,
bf16 autocast):
  adamw      : torch.optim.AdamW over all params (M-line recipe placeholder lr)
  muon_native: GroupMuon(native) on 91 matrices + aux AdamW on embed/norms
  group_A    : GroupMuon(mix on FFN=39, lam=0.75, norm_match) + native rest + aux

Measures: s/step, optimizer seconds, ns5 seconds (native vs mix), VRAM peak,
loss trajectory sanity, fp64 direction diagnostics at final step.
Writes results/smoke_100m.json (+ echo to stdout). No eval split consumption.
"""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_aware_muon.data import WindowStream, build_cache, eta_schedule
from context_aware_muon.group_muon import GroupMuon
from context_aware_muon.model_100m import build_model, param_fingerprint
from context_aware_muon.routing import route_model

CACHE = ROOT / "artifacts" / "smoke_tokens_1p2m.u32"
STEPS = 30
MICRO, ACCUM, SEQ = 8, 4, 256
PEAKS = {"adamw": 1e-3, "muon_native": 1e-2, "group_A": 1e-2}
WARMUP, TOTAL = 8, STEPS
SEED = 3471


def main():
    dev = "cuda:0"
    assert torch.cuda.is_available()
    if not CACHE.exists():
        meta = build_cache(CACHE, 1_200_000)
        print("cache built:", meta["tokens"], "tokens")
    stream = WindowStream(CACHE, SEQ)
    out = {"device": torch.cuda.get_device_name(0), "steps": STEPS,
           "tokens_per_step": MICRO * ACCUM * SEQ, "seed": SEED, "paths": {}}

    for path in ("adamw", "muon_native", "group_A"):
        torch.manual_seed(SEED)
        model = build_model("100m", seed=SEED, device=dev)
        init_fp = param_fingerprint(model)
        model.train()
        routed = route_model(model)

        opt_secs = 0.0
        ns_native = ns_mix = 0.0
        cos_g = cos_onat = None
        nerr = None
        if path == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=PEAKS[path],
                                    betas=(0.9, 0.95), eps=1e-8, weight_decay=0.02)
            muon = aux = None
        else:
            mix_params, native_params = [], []
            if path == "muon_native":
                native_params = routed["ffn"] + routed["vo"] + routed["qk"]
            else:
                mix_params = routed["ffn"]
                native_params = routed["vo"] + routed["qk"]
            peak = PEAKS[path]
            groups = []
            if mix_params:
                groups.append({"params": mix_params, "mode": "mix", "lam": 0.75,
                               "norm_match": True, "lr": peak})
            if native_params:
                groups.append({"params": native_params, "mode": "native", "lr": peak})
            muon = GroupMuon(groups, lr=peak, weight_decay=0.02, momentum=0.95,
                             nesterov=True)
            aux = torch.optim.AdamW(routed["aux"], lr=peak, betas=(0.9, 0.95),
                                    eps=1e-8, weight_decay=0.02)
            opt = None

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        losses = []
        t0 = time.perf_counter()
        cursor0 = stream.cursor
        for step in range(STEPS):
            lr = eta_schedule(step, TOTAL, WARMUP, PEAKS[path])
            if opt is not None:
                for g in opt.param_groups:
                    g["lr"] = lr
            else:
                for g in muon.param_groups:
                    g["lr"] = lr
                for g in aux.param_groups:
                    g["lr"] = lr
            tot = 0.0
            for _ in range(ACCUM):
                x, y = stream.next_batch(MICRO)
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, labels=y).loss
                (loss / ACCUM).backward()
                tot += float(loss)
            ts = time.perf_counter()
            if opt is not None:
                opt.step()
            else:
                muon.step()
                aux.step()
            opt_secs += time.perf_counter() - ts
            if muon is not None:
                ns_native += muon.stats["ns_seconds_native"]
                ns_mix += muon.stats["ns_seconds_mix"]
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            else:
                muon.zero_grad(set_to_none=True)
                aux.zero_grad(set_to_none=True)
            losses.append(tot / ACCUM)
            if step == STEPS - 1 and muon is not None:
                cos_g = muon.stats["cos_O_g_mean"]
                cos_onat = muon.stats["cos_O_Onat_mean"]
                nerr = muon.stats["norm_rel_err_max"]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out["paths"][path] = {
            "wall_s": wall,
            "s_per_step": wall / STEPS,
            "opt_step_seconds_total": opt_secs,
            "ns_seconds_native_total": ns_native,
            "ns_seconds_mix_total": ns_mix,
            "vram_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            "loss_first3": losses[:3],
            "loss_last3": losses[-3:],
            "tokens_consumed": stream.cursor - cursor0,
            "init_fingerprint": init_fp,
            "final_fingerprint": param_fingerprint(model),
            "cos_O_g_last": cos_g,
            "cos_O_Onat_last": cos_onat,
            "norm_rel_err_max_last": nerr,
        }
        print(path, json.dumps({k: v for k, v in out["paths"][path].items() if not isinstance(v, list)}))
        del model, opt
        try:
            del muon, aux
        except NameError:
            pass
        torch.cuda.empty_cache()

    out["throughput_tok_s"] = {
        p: out["paths"][p]["tokens_consumed"] / out["paths"][p]["wall_s"] for p in out["paths"]
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "smoke_100m.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["throughput_tok_s"], indent=1))


if __name__ == "__main__":
    main()
