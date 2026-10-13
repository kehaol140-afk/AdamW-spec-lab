"""Package existing V14 evidence without modifying the frozen protocol or results."""
import json
import zipfile

from mechanism_screen_v14 import screen as s


def main():
    s.verify_manifest()
    base = s.HERE
    names = [
        "screen.py", "test_screen.py", "posthoc_audit.py", "pack_light.py",
        "config.json", "RUN_VERIFIED.ps1", "RUN_COMMANDS.md",
        "run_manifest.json", "parent_dependencies.json", "candidate_list.json",
        "confirm_documents.csv", "selection_frozen.json", "dev_scores.json",
        "numerical_calibration.jsonl", "posthoc_calibration.json", "budget_ledger.json",
        "findings.json", "V14_FINDINGS.md", "tests_cpu.log",
    ]
    for pattern in ("direction_w*.json", "raw/dev/*.jsonl", "raw/confirm/*.jsonl", "logs/*"):
        names.extend(str(p.relative_to(base)).replace("\\", "/") for p in sorted(base.glob(pattern)) if p.is_file())
    for p in sorted((base / "attempts").rglob("*")):
        if p.is_file() and p.name != "confirm_views.jsonl":
            names.append(str(p.relative_to(base)).replace("\\", "/"))
    names = sorted(set(names))
    manifest = {"protocol_id": s.CFG["protocol_id"],
                "active_manifest_sha256": s.sha(base / "run_manifest.json"),
                "files": {name: s.sha(base / name) for name in names},
                "excluded": ["confirm_views.jsonl", "attempts/pre_gpu_revision_1/confirm_views.jsonl",
                             "original corpus", "large frozen checkpoint tensors"]}
    s.write(base / "light_manifest.json", manifest)
    with zipfile.ZipFile(base / "v14_light.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names + ["light_manifest.json"]:
            z.write(base / name, name)
    with zipfile.ZipFile(base / "v14_light.zip") as z:
        if set(z.namelist()) != set(names + ["light_manifest.json"]):
            raise RuntimeError("ZIP_MEMBER_MISMATCH")
        for name in names:
            if __import__("hashlib").sha256(z.read(name)).hexdigest() != manifest["files"][name]:
                raise RuntimeError("ZIP_SHA_MISMATCH:" + name)
    print(json.dumps({"members": len(names)+1, "zip_sha256": s.sha(base/"v14_light.zip"),
                      "zip_bytes": (base/"v14_light.zip").stat().st_size}))


if __name__ == "__main__":
    main()
