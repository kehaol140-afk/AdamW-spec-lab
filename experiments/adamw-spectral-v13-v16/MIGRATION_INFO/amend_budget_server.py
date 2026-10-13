#!/usr/bin/env python3
"""One-time server budget amendment for ADAMW_SPECTRAL_CAMPAIGN_V16_20261002.

Authorized by the user's migration decision (2026-10-10). Edits ONLY operational
budget fields; scientific counters (optimizer_step_attempts/completions,
token_forward_equivalents, forward_calls, backward_attempts, train_effective_tokens,
train/direction/eval/device seconds, anchor_svds) are preserved untouched.

What it does (idempotency-guarded, logged to DECISIONS.md):
  1. budget_ledger.json: wall_started_at -> now (new 72h window); pause_seconds -> 0
     (old value copied to pause_seconds_windows_session for the record).
  2. STATE.json: drop 'stop' (STOP_RESOURCE) so the serial runner may continue.
  3. Append a DECISIONS.md entry with before/after values.

Run from the project root on the server:  python3.9 MIGRATION_INFO/amend_budget_server.py
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "campaign_v16"


def main():
    ledger_path = CAMP / "budget_ledger.json"
    state_path = CAMP / "STATE.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))

    before = {"wall_started_at": ledger.get("wall_started_at"),
              "pause_seconds": ledger.get("pause_seconds"),
              "stop": state.get("stop")}
    if before["stop"] is None and time.time() - (before["wall_started_at"] or 0) < 72 * 3600:
        print("already amended; nothing to do")
        return

    old_pause = ledger.get("pause_seconds", 0.0)
    ledger["wall_started_at"] = time.time()
    ledger["pause_seconds"] = 0.0
    ledger["pause_seconds_windows_session"] = old_pause
    ledger["budget_amendment"] = {
        "authorized": "user migration decision 2026-10-10 (SERVER_MIGRATION_PLAN.md #3)",
        "fields": ["wall_started_at", "pause_seconds", "STATE.stop"],
        "scientific_counters_unchanged": True,
        "amended_at": time.time()}

    state.pop("stop", None)
    state.pop("stop_note", None)

    for path, obj in ((ledger_path, ledger), (state_path, state)):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
                       encoding="utf-8")
        tmp.replace(path)

    with (CAMP / "DECISIONS.md").open("a", encoding="utf-8") as f:
        f.write("\n## %s (server budget amendment)\n"
                "- 条件: 迁移至服务器续跑，本地 72h 墙钟与 2h 暂停上限已过\n"
                "- 证据: RAM_RECOVERY.md + SERVER_MIGRATION_PLAN.md #3；修订前 %s\n"
                "- 行动: wall_started_at 重置为本机时刻（新 72h 窗口）、pause_seconds 归零"
                "（原值记 pause_seconds_windows_session）、清除 STATE.stop；"
                "全部科学计数器与冻结阈值零改动\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(before)))
    print("amended:", json.dumps(before), "->",
          json.dumps({"wall_started_at": ledger["wall_started_at"],
                      "pause_seconds": 0.0, "stop": None}))


if __name__ == "__main__":
    main()
