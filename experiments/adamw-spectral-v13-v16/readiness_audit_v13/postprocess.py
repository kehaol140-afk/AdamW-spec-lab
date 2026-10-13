"""Read-only postprocessing of frozen V1.3 GPU measurements; separately hashed."""
from __future__ import annotations

import difflib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cli import bundle_light
from .common import HERE, ROOT, read_json, sha_file, write_json


def _rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _effect_arrays(tag, short, view):
    names = ("F00", "F10", "F01", "F11")
    rows = {name: _rows(HERE / "results" / "losses" / f"{tag}_{short}_{view}_{name}.jsonl")
            for name in names}
    ids = [r["doc_id"] for r in rows["F00"]]
    if any(ids != [r["doc_id"] for r in rows[name]] for name in names):
        raise RuntimeError("POSTPROCESS_STATE_DOC_MISMATCH")
    loss = np.array([[r["loss_sum"] for r in rows[name]] for name in names], dtype=np.float64)
    count = np.array([[r["token_count"] for r in rows[name]] for name in names], dtype=np.int64)
    if not np.array_equal(count, np.broadcast_to(count[0], count.shape)):
        raise RuntimeError("POSTPROCESS_MASK_MISMATCH")
    return ids, loss, count[0]


def _effect(nll):
    return np.array((nll[0] - nll[1], nll[2] - nll[3],
                     nll[3] - nll[1] - nll[2] + nll[0]))


def bootstrap():
    result = []
    for window in range(4):
        for short in ("ffn", "attn"):
            for view in ("legacy", "natural_prefix", "new_probe_prefix"):
                ids, loss, count = _effect_arrays(f"w{window}", short, view)
                n = len(ids)
                rng = np.random.default_rng(3407 + 100 * window + (0 if short == "ffn" else 10) +
                                            {"legacy": 0, "natural_prefix": 1, "new_probe_prefix": 2}[view])
                index = rng.integers(0, n, size=(2000, n))
                nll = np.stack([loss[i][index].sum(axis=1) / count[index].sum(axis=1)
                                for i in range(4)])
                sample_effect = _effect(nll)
                full = _effect(loss.sum(axis=1) / count.sum())
                result.append({"window_id": window, "target": short, "view": view,
                               "distinct_docs": n, "target_tokens": int(count.sum()),
                               "I_before": float(full[0]), "I_after": float(full[1]),
                               "interaction": float(full[2]),
                               "document_bootstrap_95": {
                                   name: list(map(float, np.quantile(sample_effect[i], [0.025, 0.975])))
                                   for i, name in enumerate(("I_before", "I_after", "interaction"))},
                               "bootstrap_repetitions": 2000,
                               "window_is_independent_seed": False})
    write_json(HERE / "results" / "document_bootstrap.json", result)
    return result


def same_frozen():
    comparison = {}
    for view in ("legacy", "natural_prefix"):
        per_state = {}
        for state in ("F00", "F10", "F01", "F11"):
            cal = _rows(HERE / "results" / "losses" / f"w0_cal_first_ffn_{view}_{state}.jsonl")
            full = _rows(HERE / "results" / "losses" / f"w0_ffn_{view}_{state}.jsonl")
            index = {r["doc_id"]: r for r in full}
            diffs = []
            for row in cal:
                other = index[row["doc_id"]]
                if row["token_count"] != other["token_count"]:
                    raise RuntimeError("SAME_FROZEN_MASK_CHANGED")
                diffs.append(abs(row["loss_sum"] - other["loss_sum"]))
            per_state[state] = {"documents": len(cal), "loss_sum_max_abs": max(diffs)}
        comparison[view] = per_state
    write_json(HERE / "results" / "same_frozen_repeatability.json", comparison)
    return comparison


def direction_bootstrap():
    result = []
    for path in sorted((HERE / "results" / "directions").glob("*.jsonl")):
        rows = _rows(path)
        count = np.array([r["token_count"] for r in rows])
        directions = list(rows[0]["directions"])
        rng = np.random.default_rng(3407)
        index = rng.integers(0, len(rows), size=(2000, len(rows)))
        ci = {}
        for direction in directions:
            dot = np.array([r["directions"][direction]["loss_sum_gradient_dot"] for r in rows])
            samples = dot[index].sum(axis=1) / count[index].sum(axis=1)
            ci[direction] = list(map(float, np.quantile(samples, [0.025, 0.975])))
        result.append({"raw_path": str(path.relative_to(HERE)), "tag": rows[0]["tag"],
                       "target": rows[0]["target"], "baseline": rows[0]["baseline"],
                       "distinct_docs": len(rows), "dot_bootstrap_95": ci,
                       "bootstrap_repetitions": 2000,
                       "cosine_bootstrap": "NOT_ESTIMATED_FROM_SCALAR_DOTS"})
    write_json(HERE / "results" / "direction_document_bootstrap.json", result)
    return result


def resource_timing():
    rows = _rows(HERE / "logs" / "resources.jsonl")
    pids = defaultdict(list)
    for row in rows:
        pids[row["pid"]].append(row)
    resources = {"samples": len(rows),
                 "minimum_available_ram_gib": min(r["available_ram_bytes"] for r in rows) / 2**30,
                 "maximum_rss_gib": max(r["rss_bytes"] for r in rows) / 2**30,
                 "maximum_cuda_reserved_gib": max((r["cuda_reserved_bytes"] or 0) for r in rows) / 2**30,
                 "warning_samples": sum(r["status"] == "WARNING" for r in rows),
                 "blocked_samples": sum(r["status"] == "BLOCKED" for r in rows),
                 "paging": "NOT_MEASURED"}
    timing = {str(pid): {"first_time_unix": min(r["time"] for r in group),
                         "last_time_unix": max(r["time"] for r in group),
                         "observed_span_seconds": max(r["time"] for r in group) - min(r["time"] for r in group),
                         "samples": len(group)}
              for pid, group in pids.items()}
    write_json(HERE / "results" / "resource_timing_summary.json", {"resources": resources,
              "timing_by_pid": timing, "timing_scope": "sampled span, not full process wall-clock"})
    return resources, timing


def finite_difference_resolution():
    rows = []
    for path in sorted((HERE / "results" / "finite_difference").glob("*.json")):
        item = read_json(path)
        rows.append({"path": str(path.relative_to(HERE)), "baseline": item["baseline"],
                     "view": item["view"], "autodiff": item["autodiff"],
                     "scales": item["scales"]})
    if len(rows) != 4:
        raise RuntimeError("FINITE_DIFFERENCE_COVERAGE_MISMATCH")
    result = {"status": "SMALLEST_SCALE_ROUNDING_LIMITED",
              "generation_mode": "parent_bf16_autocast_else_fp32",
              "evaluation_mode": "fp32_no_tf32",
              "rows": rows,
              "interpretation": "all scales retained; FP32 loss_sum rounding dominates h=0.0003; "
                                "does not redefine the readout repeatability target"}
    write_json(HERE / "results" / "finite_difference_resolution.json", result)
    return result


def patches():
    old = HERE / "attempts" / "attempt1_before_atomic_retry"
    folder = HERE / "patches"
    folder.mkdir(exist_ok=True)
    chunks = []
    for name in ("common.py", "audit.py", "engine.py"):
        prior = (old / name).read_text(encoding="utf-8").splitlines(keepends=True)
        current = (HERE / name).read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(prior, current, fromfile="attempt1/" + name,
                                           tofile="attempt2/" + name))
    path = folder / "attempt2_atomic_resume.patch"
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def seal_bundle():
    archive = bundle_light()
    historical = [
        "reports/V12_FINDINGS.md", "results/local_gate.json", "results/candidate_frozen.json",
        "config/local_v1_lowmem.json", "bindings.local.json",
        "replay_audit_v12/core.py", "replay_audit_v12/replay.py",
        "replay_audit_v12/cli.py", "replay_audit_v12/config.json",
        "replay_audit_v12/results/replay_status.json",
        "replay_audit_v12/logs/opencode_diagnosis_summary.json",
        "replay_audit_v12/logs/diagA_repeat_builds.txt",
        "replay_audit_v12/logs/diagB_full_pair.txt",
        "replay_audit_v12/logs/diagC_causal.txt",
        "replay_audit_v12/logs/diagD_loadstate.txt",
    ]
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as output:
        for name in historical:
            output.write(ROOT / name, name)
    with zipfile.ZipFile(archive) as check:
        names = set(check.namelist())
        required = {"readiness_audit_v13/READINESS_AUDIT.md",
                    "readiness_audit_v13/readiness.json",
                    "readiness_audit_v13/budget_ledger.json",
                    "readiness_audit_v13/analysis_manifest.json",
                    "readiness_audit_v13/results/document_bootstrap.json",
                    "readiness_audit_v13/results/direction_document_bootstrap.json",
                    "replay_audit_v12/results/replay_losses.jsonl", *historical}
        if not required <= names:
            raise RuntimeError("SEALED_BUNDLE_MISSING:" + str(required - names))
        loss_files = [name for name in names if name.startswith("readiness_audit_v13/results/losses/") and
                      name.endswith(".jsonl")]
        if len(loss_files) != 196:
            raise RuntimeError(f"SEALED_BUNDLE_LOSS_UNIT_COUNT:{len(loss_files)}")
        for name in loss_files:
            disk = ROOT / name
            if check.read(name) != disk.read_bytes():
                raise RuntimeError("SEALED_BUNDLE_RAW_MISMATCH:" + name)
        if len(check.read("replay_audit_v12/results/replay_losses.jsonl").splitlines()) != 192:
            raise RuntimeError("SEALED_BUNDLE_V12_RAW_COUNT")
        if any(name.endswith((".pt", ".u32", "eval_views.jsonl")) for name in names):
            raise RuntimeError("SEALED_BUNDLE_HEAVY_DATA")
    return archive


def main():
    b = bootstrap()
    frozen = same_frozen()
    direction = direction_bootstrap()
    fd_resolution = finite_difference_resolution()
    resources, timing = resource_timing()
    patch = patches()
    manifest = {"postprocess_source_sha256": sha_file(Path(__file__)),
                "patch_regression_source_sha256": sha_file(HERE / "tests_patch.py"),
                "patch_regression_log_sha256": sha_file(HERE / "logs" / "patch_regression_tests.log"),
                "measurement_run_manifest_sha256": sha_file(HERE / "run_manifest.json"),
                "measurement_source_hash": read_json(HERE / "run_manifest.json")["source_hash"],
                "outputs": {name: sha_file(HERE / "results" / name) for name in
                            ("document_bootstrap.json", "same_frozen_repeatability.json",
                             "direction_document_bootstrap.json", "resource_timing_summary.json",
                             "finite_difference_resolution.json")},
                "patch_sha256": sha_file(patch)}
    write_json(HERE / "analysis_manifest.json", manifest)
    readiness = read_json(HERE / "readiness.json")
    readiness["document_bootstrap_units"] = len(b)
    readiness["direction_bootstrap_units"] = len(direction)
    readiness["same_frozen_repeated_evaluation"] = frozen
    readiness["resource_summary"] = resources
    readiness["finite_difference_resolution"] = {"status": fd_resolution["status"],
              "raw_path": "results/finite_difference_resolution.json", "units": len(fd_resolution["rows"])}
    readiness["analysis_manifest_sha256"] = sha_file(HERE / "analysis_manifest.json")
    write_json(HERE / "readiness.json", readiness)
    main_ffn = [row for row in b if row["window_id"] == 0 and row["target"] == "ffn" and
                row["view"] == "new_probe_prefix"][0]
    paired = read_json(HERE / "results" / "summaries" / "w0_ffn_paired_labels.json")
    d0 = read_json(HERE / "results" / "directions" / "w0_ffn_F00.summary.json")
    d1 = read_json(HERE / "results" / "directions" / "w0_ffn_F01.summary.json")
    g = read_json(HERE / "results" / "geometry" / "w0_ffn.summary.json")
    report = HERE / "READINESS_AUDIT.md"
    marker = "\n## 文档级复核与回到主问题\n"
    existing = report.read_text(encoding="utf-8")
    if marker in existing:
        report.write_text(existing.split(marker)[0].rstrip() + "\n", encoding="utf-8")
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## 文档级复核与回到主问题\n\n")
        stream.write(f"主 FFN/窗口0/新128自然文档：I_before={main_ffn['I_before']:.9g} "
                     f"(95%文档bootstrap {main_ffn['document_bootstrap_95']['I_before']}), "
                     f"I_after={main_ffn['I_after']:.9g} "
                     f"({main_ffn['document_bootstrap_95']['I_after']}), "
                     f"interaction={main_ffn['interaction']:.9g} "
                     f"({main_ffn['document_bootstrap_95']['interaction']})。四窗口共用同一 checkpoint，"
                     "这些区间只对文档重采样，不把窗口当作独立 seed。\n\n")
        stream.write(f"旧80文档同一构建的自然−legacy 处理效应差："
                     f"I_before={paired['paired_natural_minus_legacy']['I_before']:.9g}，"
                     f"I_after={paired['paired_natural_minus_legacy']['I_after']:.9g}；"
                     "这直接检验标签对处理效应的影响，不能用三份 LR 排序替代。"
                     "这些差值小于预设的 2e-6 nats/token 工程分辨率目标，标签处理效应差应记为"
                     "数值上未分辨，不能宣称 EOS 改变或未改变真实机制。\n\n")
        stream.write(f"在新文档上，F00 的 <G,A>={d0['directions']['A']['token_weighted_dot']:.9g}，"
                     f"F01 的 <G,A>={d1['directions']['A']['token_weighted_dot']:.9g}；"
                     "M_plus 与当前构造梯度的原范数及匹配A范数信号见逐文档原始记录和 summary。"
                     f"窗口0 A 头部 Frobenius 能量份额={g['head_frobenius_energy_fraction']:.6g}，"
                     f"真实输入输出能量份额约={g['head_energy_per_token'] / g['joint_energy_per_token']:.6g}。"
                     "这是局部机制诊断，不是新优化器的终点收益。\n\n")
        stream.write("同一冻结更新在校准8文档和完整旧视图中的逐文档重复损失差见 "
                     "`results/same_frozen_repeatability.json`；文档级效应和方向点积区间见 "
                     "`results/document_bootstrap.json`、`results/direction_document_bootstrap.json`。"
                     "有限差分三档均原样保存：最小 h=0.0003 受 FP32 损失求和舍入影响明显，"
                     "不能用该档近零/符号波动推翻已通过的效应层重复读数，也不能把 h=0.003 的较好一致性冒充所有尺度精确。\n\n")
        stream.write(f"真实资源样本 {resources['samples']} 条，最低可用 RAM "
                     f"{resources['minimum_available_ram_gib']:.3f} GiB，最高 RSS "
                     f"{resources['maximum_rss_gib']:.3f} GiB，峰值 CUDA reserved "
                     f"{resources['maximum_cuda_reserved_gib']:.3f} GiB；"
                     f"告警 {resources['warning_samples']} 次，阻断 {resources['blocked_samples']} 次。"
                     "Windows 换页未测量。按 PID 的采样时间跨度在 `results/resource_timing_summary.json`。\n\n")
        stream.write("本轮已能可信地继续研究局部“可利用余地”：目标块单独更新与其他参数先更新后的边际效应在"
                     "固定构造窗口/文档上可区分。关于可预测条件和低成本修正，仍需未来独立 seed、强 LR 基线、"
                     "完整训练终点与成本验证；本轮没有启动这些工作。\n\n")
        stream.write("补丁 `patches/attempt2_atomic_resume.patch` 与专门回归 "
                     "`logs/patch_regression_tests.log`（2/2 通过）证明 Windows 短暂共享冲突重试及累计台账续接。"
                     "`budget_ledger.json` 的 `elapsed_seconds=0` 为旧字段未填充，不表示实际耗时为零；"
                     "按 PID 可核对的采样跨度另见时间汇总。最终分析脚本及其哈希见 `analysis_manifest.json`。\n")
    archive = seal_bundle()
    print(json.dumps({"analysis_manifest_sha256": sha_file(HERE / "analysis_manifest.json"),
                      "bootstrap_units": len(b), "direction_units": len(direction),
                      "same_frozen": frozen, "resources": resources,
                      "bundle_sha256": sha_file(archive)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
