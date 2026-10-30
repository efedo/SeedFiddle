"""Check or repair the PlantCV reference lock's pinned CRLF byte representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOCKFILE = Path("config/plantcv-reference-requirements.txt")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_lock_details() -> tuple[Path, str]:
    config_path = ROOT / "config" / "peer_reference_sources.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        environment = config["environment"]
        lockfile = Path(environment["lockfile"])
        expected = str(environment["lockfile_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read the PlantCV lock configuration {config_path}: {error}") from error

    if lockfile != EXPECTED_LOCKFILE:
        raise ValueError(
            f"Configured PlantCV lock path must be {EXPECTED_LOCKFILE.as_posix()}, "
            f"not {lockfile}."
        )
    return ROOT / lockfile, expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="rewrite the lock only when CRLF conversion matches its existing checksum",
    )
    args = parser.parse_args(argv)

    try:
        lock_path, expected = _load_lock_details()
        actual_bytes = lock_path.read_bytes()
    except (OSError, ValueError) as error:
        print(f"Reference lock error: {error}", file=sys.stderr)
        return 1

    actual = _sha256(actual_bytes)
    print(f"Actual SHA-256:   {actual}")
    print(f"Expected SHA-256: {expected}")
    if actual == expected:
        print("Reference lock checksum matches; no change needed.")
        return 0

    candidate = actual_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    candidate_sha = _sha256(candidate)
    if not args.repair:
        if candidate_sha == expected:
            print("Mismatch found; CRLF line-ending conversion would resolve it.")
        else:
            print(
                "Mismatch found; CRLF line-ending conversion would not resolve it "
                f"(candidate SHA-256: {candidate_sha})."
            )
        return 1

    if candidate_sha != expected:
        print(
            "Refusing repair: CRLF conversion does not match the pinned checksum "
            f"(candidate SHA-256: {candidate_sha}). The file was not changed.",
            file=sys.stderr,
        )
        return 1

    try:
        lock_path.write_bytes(candidate)
        written_sha = _sha256(lock_path.read_bytes())
    except OSError as error:
        print(f"Reference lock repair failed: {error}", file=sys.stderr)
        return 1
    print(f"Written SHA-256:  {written_sha}")
    if written_sha != expected:
        print(
            "Reference lock repair verification failed; expected checksum was not written.",
            file=sys.stderr,
        )
        return 1
    print("Reference lock repaired to the pinned CRLF byte representation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
