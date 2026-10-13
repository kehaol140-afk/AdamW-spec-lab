"""Read-only postmeasurement checks. This file did not define evaluated candidates."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch

from mechanism_screen_v14 import screen as s


def main():
    m = s.verify_manifest()
    l = s.ledger()
    selected = s.read(s.HERE / "selection_frozen.json")
    resources = [json.loads(x) for x in (s.HERE / "logs/resources.jsonl").open(encoding="utf-8")]
    timings = [json.loads(x) for x in (s.HERE / "logs/timing.jsonl").open(encoding="utf-8")]
    phase_counts = {}
    for phase, candidate_ids in (("dev", [x["id"] for x in s.candidates()]),
                                 ("confirm", list(dict.fromkeys([*selected["selected"].values(), "scalar_g1_s1"])))):
        tokens = count = 0
        ids_ref = None
        for w in range(4):
            for cid in candidate_ids:
                rows = [json.loads(line) for line in s._unit_path(phase,w,cid).open(encoding="utf-8")]
                if len(rows) != 128:
                    raise RuntimeError("RAW_UNIT_INCOMPLETE")
                ids = [(r["doc_id"],r["token_count"]) for r in rows]
                if ids_ref is None:
                    ids_ref = ids
                if ids != ids_ref:
                    raise RuntimeError("RAW_PAIRING_OR_MASK_CHANGED")
                tokens += sum(r["token_count"] for r in rows)
                count += len(rows)
        phase_counts[phase] = {"rows":count,"token_forward_equivalents":tokens,
                               "unique_docs":len({doc for doc,_ in ids_ref})}
    if sum(x["rows"] for x in phase_counts.values()) != l["forward_attempts"]:
        raise RuntimeError("FORWARD_LEDGER_COUNT_MISMATCH")
    if sum(x["token_forward_equivalents"] for x in phase_counts.values()) != l["token_forward_equivalents"]:
        raise RuntimeError("FORWARD_LEDGER_TOKEN_MISMATCH")
    if l["optimizer_step_attempts"] or l["backward_attempts"]:
        raise RuntimeError("UNEXPECTED_UPDATE_OR_BACKWARD")
    beta1,beta2=s.PCFG["baseline"]["betas"]
    eps=s.PCFG["baseline"]["eps"]
    moment_rows=[]
    for w,path in enumerate(s.FROZEN):
        f=torch.load(path,map_location="cpu",weights_only=False)
        target=s.TARGET
        a=f["A"][target].double()
        mp=f["M_plus"][target].double()
        vp=f["moments_after"][target]["exp_avg_sq"].double()
        step=int(f["moments_after"][target]["step"])
        reconstructed=(mp/(1-beta1**step))/(torch.sqrt(vp/(1-beta2**step))+eps)
        max_abs=float((reconstructed-a).abs().max())
        moment_rows.append({"window":w,"step_after":step,"max_abs_A_vs_Mplus_Vplus":max_abs,
                            "frozen_sha256":s.sha(path)})
        if max_abs > f["formula_tolerance"]:
            raise RuntimeError("FROZEN_MOMENT_DIRECTION_MISMATCH")
        del f
    chosen = selected["selected"]
    confirm = {cid: [[json.loads(line) for line in s._unit_path("confirm",w,cid).open(encoding="utf-8")]
                     for w in range(4)] for cid in chosen.values()}
    n = 128
    def nll(cid, indices):
        return sum(sum(confirm[cid][w][i]["loss_sum"] for i in indices) /
                   sum(confirm[cid][w][i]["token_count"] for i in indices) for w in range(4))/4
    rng = np.random.default_rng(s.CFG["bootstrap_seed"])
    fresh_direct = {}
    for family in ("fresh_gradient", "fresh_preconditioned"):
        point = nll(chosen["spectral"],range(n))-nll(chosen[family],range(n))
        samples=[]
        for _ in range(s.CFG["bootstrap_repetitions"]):
            indices=rng.integers(0,n,n)
            samples.append(nll(chosen["spectral"],indices)-nll(chosen[family],indices))
        lo,hi=np.quantile(samples,[.025,.975])
        fresh_direct[family]={"spectral_minus_fresh_nll":point,"ci95_lower":float(lo),"ci95_upper":float(hi)}
    audit={"protocol_id":s.CFG["protocol_id"],"manifest_sha256":s.sha(s.HERE/"run_manifest.json"),
           "parent_dependencies_sha256":s.sha(s.HERE/"parent_dependencies.json"),
           "selection_created_at":selected["created_at"],
           "first_confirm_resource_time":min(x["time"] for x in resources if x["phase"].startswith("confirm_")),
           "phase_counts":phase_counts,"ledger":l,"moment_direction_calibration":moment_rows,
           "direct_nonspectral_comparisons":fresh_direct,
           "resources":{"minimum_available_ram_bytes":min(x["available_ram_bytes"] for x in resources),
                        "maximum_rss_bytes":max(x["rss_bytes"] for x in resources),
                        "maximum_cuda_reserved_bytes":max(x["cuda_reserved_bytes"] or 0 for x in resources),
                        "warning_rows":sum(x["status"]=="WARNING" for x in resources),
                        "blocked_rows":sum(x["status"]=="BLOCKED" for x in resources)},
           "time":{"gpu_evaluation_seconds":sum(x["seconds"] for x in timings),
                   "svd_seconds":l["svd_seconds"],
                   "elapsed_since_first_ledger_start_seconds":max(x["time"] for x in resources)-l["started_at"]},
           "source_lineage":{"previous_manifest_sha256":m.get("previous_manifest_sha256"),
                             "retry_archive":m.get("previous_attempt_archive"),
                             "prepatch_raw_sha256":m.get("prepatch_raw_sha256")}}
    if audit["first_confirm_resource_time"] <= audit["selection_created_at"]:
        raise RuntimeError("CONFIRM_OPENED_BEFORE_SELECTION")
    s.write(s.HERE/"posthoc_calibration.json",audit)
    print(json.dumps({"phase_counts":phase_counts,"moment_max_abs":max(x["max_abs_A_vs_Mplus_Vplus"] for x in moment_rows)},ensure_ascii=False))


if __name__=="__main__":
    main()
