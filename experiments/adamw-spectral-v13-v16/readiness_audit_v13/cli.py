"""V1.3 CLI; no parent write operation is exposed."""
from __future__ import annotations

import argparse
import json
import traceback
import zipfile
from pathlib import Path

from .audit import prepare, verify_after, verify_inputs
from .common import HERE, ROOT, V12, read_json, sha_file, source_hash, write_json


def _summary_files():
    folder = HERE / "results" / "summaries"
    return list(folder.glob("*.json")) if folder.exists() else []


def finalize():
    blockers = []
    try:
        verify_inputs()
        verify_after()
        parent_unchanged = True
    except BaseException as exc:
        parent_unchanged = False
        blockers.append("PROTECTED_INPUT_CHANGED:" + str(exc))
    ledger = read_json(HERE / "budget_ledger.json") if (HERE / "budget_ledger.json").exists() else None
    summaries = [read_json(path) for path in _summary_files()]
    full = [r for r in summaries if r.get("tag", "").startswith("w") and
            "cal_" not in r.get("tag", "") and r.get("tag") != "w0_full_repeat" and
            r.get("view") in ("legacy", "natural_prefix", "new_probe_prefix", "new_probe_packed")]
    coverage = {}
    for result in full:
        coverage.setdefault(str(result["window_id"]), {}).setdefault(result["target"], {})[result["view"]] = {
            "documents": result["documents"], "target_tokens": result["target_tokens"],
            "I_before": result["I_before"], "I_after": result["I_after"],
            "interaction": result["interaction"]}
    local = read_json(HERE / "results" / "calibration_local.json") if (HERE / "results" / "calibration_local.json").exists() else None
    cross = read_json(HERE / "results" / "calibration_cross.json") if (HERE / "results" / "calibration_cross.json").exists() else None
    w0full = read_json(HERE / "results" / "w0_full_repeatability.json") if (HERE / "results" / "w0_full_repeatability.json").exists() else None
    numeric = {"local": local["numeric_resolution"] if local else None,
               "cross_process": cross["numeric_resolution"] if cross else None,
               "window0_full": w0full}
    numeric_pass = all(item and item["pass"] for item in numeric.values())
    directions = list((HERE / "results" / "directions").glob("*.summary.json")) if (HERE / "results" / "directions").exists() else []
    geometry = list((HERE / "results" / "geometry").glob("*.summary.json")) if (HERE / "results" / "geometry").exists() else []
    paired = list((HERE / "results" / "summaries").glob("*_paired_labels.json")) if (HERE / "results" / "summaries").exists() else []
    fd = list((HERE / "results" / "finite_difference").glob("*.json")) if (HERE / "results" / "finite_difference").exists() else []
    expected_full = 4 * 2 * 4
    expected_directions = 4 * 2 * 2
    expected_geometry = 4 * 2
    complete = (len(full) == expected_full and len(directions) == expected_directions and
                len(geometry) == expected_geometry and len(paired) == 8 and len(fd) == 4)
    if not numeric_pass:
        blockers.append("NUMERIC_RESOLUTION_INCOMPLETE_OR_FAILED")
    if not complete:
        blockers.append(f"COVERAGE_INCOMPLETE:four_state={len(full)}/{expected_full},"
                        f"direction={len(directions)}/{expected_directions},"
                        f"geometry={len(geometry)}/{expected_geometry},paired={len(paired)}/8,FD={len(fd)}/4")
    if ledger and (ledger["token_forward_equivalents"] > 5000000 or
                   ledger["gpu_optimizer_step_attempts"] > 16):
        blockers.append("BUDGET_OVER_LIMIT")
    if not ledger:
        blockers.append("NO_V13_BUDGET_LEDGER")
    if not parent_unchanged or "BUDGET_OVER_LIMIT" in blockers:
        status = "BLOCKED"
    elif complete and numeric_pass:
        status = "READY_FOR_MECHANISM_STUDY"
    elif full and numeric_pass:
        status = "PARTIAL_READY"
    else:
        status = "BLOCKED"
    old_gate = read_json(ROOT / "results" / "local_gate.json")
    readiness = {
        "protocol_id": read_json(HERE / "config.json")["protocol_id"],
        "evidence_integrity": "VERIFIED" if parent_unchanged else "FAILED",
        "state_isolation_and_step_parity": {
            "cpu_tests": "see logs/cpu_tests.log", "frozen_builds": len(list((HERE / "frozen").glob("*.json"))) if (HERE / "frozen").exists() else 0,
            "stage_comparisons": len(list((HERE / "results").glob("compare_*.json"))) if (HERE / "results").exists() else 0},
        "repeatability_scope": "FFN/window0/first8_old_probe_select plus full views" if w0full else "not_full",
        "generation_mode": "parent_bf16_autocast_else_fp32",
        "evaluation_mode": "fp32_no_tf32",
        "numeric_resolution": numeric,
        "four_state_coverage": coverage,
        "gradient_geometry_coverage": {"direction_units": len(directions), "geometry_units": len(geometry),
                                       "finite_difference_units": len(fd)},
        "budget_reconciled": (HERE / "history_reconciliation.json").exists() and ledger is not None,
        "new_budget": ledger,
        "historical_budget": read_json(HERE / "history_reconciliation.json") if (HERE / "history_reconciliation.json").exists() else None,
        "parent_unchanged": parent_unchanged,
        "parent_gate": {"verdict": old_gate["verdict"], "next_action": old_gate["next_action"],
                        "primary_status": old_gate["primary_statistics"]["status"]},
        "baseline_tuning": "UNRESOLVED", "method_novelty": "NOT_ESTABLISHED",
        "end_to_end_gain": "NOT_TESTED", "deployment_cost": "NOT_TESTED",
        "formal_training_authorized": False,
        "overall_status": status, "remaining_blockers": blockers,
        "new_formal_pretraining_steps": 0,
    }
    write_json(HERE / "readiness.json", readiness)
    _write_report(readiness, full, paired, directions, geometry, fd)
    path = bundle_light()
    return {"overall_status": status, "four_state_units": len(full),
            "new_gpu_step_attempts": ledger["gpu_optimizer_step_attempts"] if ledger else 0,
            "new_token_forward_equivalents": ledger["token_forward_equivalents"] if ledger else 0,
            "bundle": str(path), "blockers": blockers}


def _write_report(readiness, full, paired, directions, geometry, fd):
    history = readiness["historical_budget"]
    budget = readiness["new_budget"]
    lines = ["# V1.3 研究准备审计", "", f"总状态：**{readiness['overall_status']}**。父 gate 保持 `BASELINE_TUNING_UNRESOLVED / STOP / NO_LOCAL_SIGNAL`；正式预训练新增 0 步。", "",
             "## 证据身份与旧结论边界", "",
             "父 source/config/data/binding、缓存、CSV 与四份 checkpoint 经 `parent_dependencies.json` 严格核验，运行前后受保护文件 SHA 清单见 `protected_before.json`/`protected_after.json`。旧三份 step512 checkpoint 在 32 个 lr_select 文档上的 192 行原始结果可复算；自然标签保持旧 LR 网格排序，但这不回答 step256 的 s=0 处理效应。diagA 与 diagC 使用不同参数遍历顺序，跨进程整体 SHA 不可作为数值变化证据。diagB 的约 8.19e-6 参数差是旧日志事实，单次 AdamW 公式约 2.98e-8 一致；阶段定位见本轮 `compare_*.json`。", "",
             "## 数值验收", "",
             f"生成模式 `{readiness['generation_mode']}`，评估模式 `{readiness['evaluation_mode']}`。2e-6 nats/token 是本轮工程分辨率目标，不是严格误差上界或新科学门槛。"]
    for name, result in readiness["numeric_resolution"].items():
        lines.append(f"- {name}: " + (f"observed_max={result.get('observed_max')}, pass={result.get('pass')}" if result else "未完成"))
    lines += ["", "## 四状态与配对标签", "",
              "所有 I 按逐文档 `loss_sum/count` 汇总：I_before=F00−F10，I_after=F01−F11，interaction=I_before−I_after。不同窗口是构造 batch，不是独立 seed；旧 probe_confirm 为已使用探索材料。"]
    for result in sorted(full, key=lambda r: (r["window_id"], r["target"], r["view"])):
        lines.append(f"- w{result['window_id']} {result['target']} {result['view']}: docs={result['documents']}, tokens={result['target_tokens']}, I_before={result['I_before']:.9g}, I_after={result['I_after']:.9g}, interaction={result['interaction']:.9g}。")
    if not full:
        lines.append("- 未获得可信的全视图四状态结果。")
    lines += ["", "同一文档 legacy/natural 的处理效应差见 `results/summaries/*_paired_labels.json`，配对文档 bootstrap 同时保留 token 权重；不能只比较绝对 NLL。"]
    for path in paired:
        obj = read_json(path)
        if obj["tag"] == "w0" and obj["target"].endswith("mlp.down_proj.weight"):
            lines.append(f"- 主 FFN/window0 自然−legacy：I_before={obj['paired_natural_minus_legacy']['I_before']:.9g}, I_after={obj['paired_natural_minus_legacy']['I_after']:.9g}, interaction={obj['paired_natural_minus_legacy']['interaction']:.9g}。")
    lines += ["", "## 方向与实际输入能量", "",
              f"方向单位 {len(directions)}/16，几何单位 {len(geometry)}/8，有限差分单位 {len(fd)}/4。方向逐文档贡献、A/M_plus/构造梯度与 A 头/剩余的 token 加权统计见 `results/directions/`；F01 真实输入下头/剩余/交叉/联合输出能量见 `results/geometry/`。谱参数能量与真实输入功能能量分别标注。"]
    lines += ["", "## 预算、资源与未解决问题", "",
              f"V1.2 原台账：{history['v12_native_ledger']['gpu_optimizer_calls_attempted']} 次 GPU step 尝试、{history['v12_native_ledger']['token_forward_equivalents']} token-forward-equivalents；diagA 的 3 次无 step 梯度构建另据脚本/日志补列至少 6144，原账本未改。",
              f"V1.3 新台账：{budget['gpu_optimizer_step_attempts'] if budget else 0}/16 次 GPU step 尝试、{budget['token_forward_equivalents'] if budget else 0}/5,000,000 token-forward-equivalents；完整原始台账为 `budget_ledger.json`。资源曲线见 `logs/resources.jsonl`；门禁为启动前 1.5 GiB、运行告警 1.5 GiB、硬停 1.0 GiB、RSS 4 GiB 与父显存约束。",
              "packed 105 窗口共享来源并连成 1 簇，仅描述性；不伪造独立文档区间。", "",
              "保留 baseline_tuning=UNRESOLVED、method_novelty=NOT_ESTABLISHED、end_to_end_gain=NOT_TESTED、deployment_cost=NOT_TESTED。仍未检验跨 seed 终点收益、强 LR 基线或低成本持续修正。", "",
              "阻断：" + ("；".join(readiness["remaining_blockers"]) if readiness["remaining_blockers"] else "无本轮工程阻断"), "",
              "运行命令与解释器固定在 `RUN_VERIFIED.ps1`；测试/运行原始日志在 `logs/`。轻量包含活动代码、配置、依赖、原始数值、最终预算和报告，排除原文、token 正文、大 checkpoint。"]
    (HERE / "READINESS_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bundle_light():
    archive_path = HERE / "readiness_v13_light.zip"
    files = [p for p in HERE.rglob("*") if p.is_file() and
             p != archive_path and "__pycache__" not in p.parts and
             ("frozen" not in p.parts or p.suffix == ".json") and p.name != "run.lock" and
             not p.name.endswith(".partial")]
    historical = [V12 / "budget_ledger.json", V12 / "evaluation_manifest.json",
                  V12 / "parent_dependencies.json", V12 / "parent_reanalysis.json",
                  V12 / "results" / "lr_reeval.json", V12 / "results" / "replay_losses.jsonl",
                  V12 / "logs" / "diag_outputs.txt"]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
        for path in historical:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(archive_path) as check:
        names = set(check.namelist())
        required = {"readiness_audit_v13/READINESS_AUDIT.md", "readiness_audit_v13/readiness.json",
                    "readiness_audit_v13/budget_ledger.json", "readiness_audit_v13/run_manifest.json",
                    "readiness_audit_v13/parent_dependencies.json", "readiness_audit_v13/RUN_VERIFIED.ps1"}
        if not required <= names:
            raise RuntimeError("LIGHT_BUNDLE_MISSING:" + str(required - names))
        if any(name.endswith((".pt", ".u32", "eval_views.jsonl")) for name in names):
            raise RuntimeError("LIGHT_BUNDLE_CONTAINS_HEAVY_OR_TOKEN_DATA")
        if len(check.read("replay_audit_v12/results/replay_losses.jsonl").splitlines()) != 192:
            raise RuntimeError("LIGHT_BUNDLE_HISTORICAL_RAW_ROW_MISMATCH")
        for name in names:
            if name.startswith("readiness_audit_v13/results/losses/") and name.endswith(".jsonl"):
                if not check.read(name).strip():
                    raise RuntimeError("LIGHT_BUNDLE_EMPTY_RESULT:" + name)
    return archive_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "calibrate-local", "calibrate-cross", "measure", "finalize", "verify"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare()
        elif args.command == "verify":
            result = {"parent_v12": "VERIFIED" if verify_inputs() else "FAILED", "source_hash": source_hash()}
        elif args.command == "finalize":
            result = finalize()
        else:
            from .engine import calibrate_local, calibrate_cross, measure
            result = {"calibrate-local": calibrate_local,
                      "calibrate-cross": calibrate_cross, "measure": measure}[args.command]()
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    except BaseException as exc:
        (HERE / "logs").mkdir(exist_ok=True)
        with (HERE / "logs" / "last_failure.txt").open("a", encoding="utf-8") as stream:
            stream.write(args.command + ": " + traceback.format_exc() + "\n")
        raise


if __name__ == "__main__":
    main()
