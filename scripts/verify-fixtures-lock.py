#!/usr/bin/env python3
"""Verify that all fixture files in fixtures.lock.json match their recorded SHA-256 hashes.

Can be used by any repository that maintains a local copy of fixtures.lock.json.

Usage:
    python scripts/verify-fixtures-lock.py
    python scripts/verify-fixtures-lock.py --lock-file /path/to/fixtures.lock.json

Exit codes:
    0  all hashes match
    1  one or more fixtures missing or drifted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--lock-file",
        default=str(REPO_ROOT / "fixtures/fixtures.lock.json"),
        metavar="FILE",
        help="path to fixtures.lock.json (default: fixtures/fixtures.lock.json)",
    )
    ap.add_argument("--json", action="store_true", help="output JSON report")
    args = ap.parse_args()

    lock_path = Path(args.lock_file)
    if not lock_path.exists():
        print(f"[error] lock file not found: {lock_path}", file=sys.stderr)
        return 1

    lock = json.loads(lock_path.read_text())
    fixtures = lock.get("fixtures", [])

    results = []
    n_ok = 0
    for entry in fixtures:
        fixture_path = REPO_ROOT / entry["fixture"]
        expected = entry["sha256"]
        if not fixture_path.exists():
            results.append({"fixture": entry["fixture"], "status": "MISSING", "expected": expected[:16]})
            continue
        actual = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        if actual == expected:
            results.append({"fixture": entry["fixture"], "status": "OK"})
            n_ok += 1
        else:
            results.append({
                "fixture": entry["fixture"],
                "status": "DRIFT",
                "expected": expected[:16],
                "actual": actual[:16],
            })

    if args.json:
        print(json.dumps({"n_total": len(fixtures), "n_ok": n_ok, "results": results}, indent=2))
    else:
        for r in results:
            if r["status"] == "OK":
                print(f"  [OK]     {r['fixture']}")
            elif r["status"] == "MISSING":
                print(f"  [MISSING] {r['fixture']}")
            else:
                print(f"  [DRIFT]  {r['fixture']}  expected={r['expected']}...  got={r['actual']}...")
        if n_ok == len(fixtures):
            print(f"\n✓ {n_ok}/{len(fixtures)} fixture hashes verified")
        else:
            print(f"\n✗ {len(fixtures) - n_ok}/{len(fixtures)} fixtures missing or drifted")
            print("  Run: python scripts/export-schemas.py  to regenerate schema fixtures")
            print("  For golden fixtures: regenerate with scripts/generate_golden_fixtures.py")

    return 0 if n_ok == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
