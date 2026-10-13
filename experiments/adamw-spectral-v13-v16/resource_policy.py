"""Small shared RAM policy; importing this module never initializes CUDA."""
from __future__ import annotations


def resource_decision(cfg: dict, available: int | None, admission: bool) -> tuple[str, str | None]:
    runtime = cfg["runtime"]
    hard = int(runtime["system_ram_free_reserve_gib"] * 2 ** 30)
    start = int(runtime.get("system_ram_start_min_gib", runtime["system_ram_free_reserve_gib"]) * 2 ** 30)
    warning = int(runtime.get("system_ram_warning_gib", runtime["system_ram_free_reserve_gib"]) * 2 ** 30)
    threshold = start if admission else hard
    if available is None or available < threshold:
        return "RESOURCE_BLOCKED", f"RAM available={available}, required={threshold}, phase={'preinit' if admission else 'runtime'}"
    if not admission and available < warning:
        return "WARNING", f"RAM available={available}, warning={warning}"
    return "READY", None
