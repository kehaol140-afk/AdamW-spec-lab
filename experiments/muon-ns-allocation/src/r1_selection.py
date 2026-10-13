"""R1 selection: near-best sets, tie keys, one-shot boundary extension, 9-role lock."""
from __future__ import annotations

import json
import math
from pathlib import Path


def tie_key(c):
    return (abs(math.log(c["a"])) + abs(math.log(c["b"])),
            abs(math.log(c["b"])),
            abs(math.log(c["a"])),
            c["a"], c["b"], c["id"])


def select_family(rows_for_state, family_filter, native_mean):
    """rows_for_state: list of {point_id, a, b, structure, V_mean}. Returns selected
    candidate (min mean_V; near-best = <= m + 1e-6; tie -> closest to native)."""
    cands = [r for r in rows_for_state if family_filter(r)]
    if not cands:
        return None
    m = min(r["V_mean"] for r in cands)
    near = [r for r in cands if r["V_mean"] <= m + 1e-6]
    return sorted(near, key=tie_key)[0], near


def boundary_touch(near, a_set, b_set, family):
    a_min, a_max = min(a_set), max(a_set)
    b_min, b_max = min(b_set), max(b_set)
    t = set()
    for r in near:
        if r["a"] == a_min:
            t.add("a_low")
        if r["a"] == a_max:
            t.add("a_high")
        if family in ("A", "S"):
            if r["b"] == b_min:
                t.add("b_low")
            if r["b"] == b_max:
                t.add("b_high")
    return t


def compute_extension(cal_by_state, A0, B0):
    """One-shot extension from INITIAL calibration only. cal_by_state: state -> rows
    (with V_mean). Returns A_final/B_final sets + trigger record."""
    triggers = {}
    touch_union = set()
    for state, rows in cal_by_state.items():
        fams = {
            "G": lambda r: r["structure"] == "NS5" and r["b"] == 1.0,
            "A": lambda r: r["structure"] == "NS5",
            "S": lambda r: r["structure"] == "NS3_MATCHED",
        }
        for fam, filt in fams.items():
            sel = select_family(rows, filt, None)
            if sel is None:
                continue
            _, near = sel
            t = boundary_touch(near, A0, B0, fam)
            triggers[f"{state}|{fam}"] = sorted(t)
            touch_union |= t
    A_final, B_final = set(A0), set(B0)
    if "a_low" in touch_union:
        A_final.add(0.25)
    if "a_high" in touch_union:
        A_final.add(1.5)
    if "b_low" in touch_union:
        B_final.add(0.5)
    if "b_high" in touch_union:
        B_final.add(1.5)
    return sorted(A_final), sorted(B_final), triggers, sorted(touch_union)


def lock_roles(cal_by_state, A_final, B_final):
    """Lock the 9 roles from FINAL calibration means per state (stores a/b/structure)."""
    lock = {}
    for state, rows in cal_by_state.items():
        def best(filt):
            cands = [r for r in rows if filt(r)]
            if not cands:
                return None
            return sorted(cands, key=lambda r: (r["V_mean"],) + tie_key(r))[0]
        G = best(lambda r: r["structure"] == "NS5" and r["b"] == 1.0)
        A = best(lambda r: r["structure"] == "NS5")
        S = best(lambda r: r["structure"] == "NS3_MATCHED")
        S_at_A = next((r for r in rows if r["structure"] == "NS3_MATCHED"
                       and r["a"] == A["a"] and r["b"] == A["b"]), None)
        A_at_S = next((r for r in rows if r["structure"] == "NS5"
                       and r["a"] == S["a"] and r["b"] == S["b"]), None)
        G_at_A = next((r for r in rows if r["structure"] == "NS5"
                       and r["a"] == A["a"] and r["b"] == 1.0), None)
        def ent(r, extra=None):
            if r is None:
                return None
            d = {"id": r["id"], "a": r["a"], "b": r["b"], "structure": r["structure"],
                 "cal_V_mean": r["V_mean"]}
            if extra:
                d.update(extra)
            return d
        lock[state] = {
            "N": {"id": "a1.0|b1.0|NS5", "a": 1.0, "b": 1.0, "structure": "NS5"},
            "G": ent(G), "A": ent(A), "S_at_A": ent(S_at_A), "S": ent(S),
            "A_at_S": ent(A_at_S), "G_at_A": ent(G_at_A),
            "Z": {"id": "Z_NO_MUON_DATA", "kind": "Z"},
            "P": {"id": "P_PRE_STATE", "kind": "P"}}
    return lock
