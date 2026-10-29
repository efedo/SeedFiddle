"""Delete stale generated test-run directories under artifacts/.

Only ``artifacts/test-runs`` and ``artifacts/test-temp`` are candidates; the
reviewed fixtures and frozen baselines elsewhere under ``artifacts/`` are never
touched.  The default is a dry run.

    .venv\\Scripts\\python.exe scripts/prune_artifacts.py --older-than 7 --delete
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRECTORIES = ("artifacts/test-runs", "artifacts/test-temp")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--older-than", type=float, default=7.0, metavar="DAYS",
                        help="keep run directories modified within this many days (default 7)")
    parser.add_argument("--delete", action="store_true",
                        help="actually delete; without it the script only lists candidates")
    args = parser.parse_args(argv)
    cutoff = time.time() - args.older_than * 86400.0
    removed_bytes = 0
    for relative in CANDIDATE_DIRECTORIES:
        parent = ROOT / relative
        if not parent.is_dir():
            continue
        for run in sorted(parent.iterdir()):
            if not run.is_dir() or run.stat().st_mtime > cutoff:
                continue
            size = _directory_bytes(run)
            removed_bytes += size
            print(f"{'delete' if args.delete else 'would delete'} {run.relative_to(ROOT)} ({size / 1e6:.1f} MB)")
            if args.delete:
                shutil.rmtree(run)
    print(f"{'freed' if args.delete else 'would free'} {removed_bytes / 1e6:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
