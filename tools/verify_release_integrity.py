#!/usr/bin/env python3
"""Verify manifest SHA256 for the release without relying on GPU/network."""
from pathlib import Path
import json, hashlib, sys

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
entries = manifest['files']
bad = []
for rel, info in entries.items():
    p = root / rel
    if not p.is_file():
        bad.append(f'MISSING {rel}')
        continue
    if p.stat().st_size != info['bytes']:
        bad.append(f'SIZE {rel}')
    if hashlib.sha256(p.read_bytes()).hexdigest() != info['sha256']:
        bad.append(f'SHA {rel}')
if len(entries) != manifest['n_files'] or sum(v['bytes'] for v in entries.values()) != manifest['total_bytes']:
    bad.append('ROOT MANIFEST COUNTS')
if 'manifest.json' in entries:
    bad.append('ROOT SELF-REFERENCE')

n_dir = 0
for mf in root.rglob('MANIFEST.sha256'):
    n_dir += 1
    for line in mf.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            bad.append(f'BAD MANIFEST LINE {mf.relative_to(root)}')
            continue
        h, name = parts
        p = root / name.lstrip('*').removeprefix('./')
        if p == mf:
            bad.append(f'SELF REFERENCE {mf.relative_to(root)}')
        elif not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(f'DIR SHA {mf.relative_to(root)} -> {name}')

missing = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - set(entries) - {'manifest.json'}
if missing:
    bad.append('UNLISTED: ' + repr(sorted(missing)[:10]))
print(f'Root entries={len(entries)}, directory manifests={n_dir}, failures={len(bad)}')
for x in bad[:30]:
    print(x)
sys.exit(bool(bad))
