"""Verify the Lane 11 archive bundle without trusting extracted filenames.

Usage: python verify_recovery.py <original_archives.tar.xz> <WAVE_ARCHIVES.sha256>
This is an artifact-integrity check, NOT scientific validation.
"""

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path


def check(bundle: Path, manifest: Path) -> dict:
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, byte_count, name = line.split()
        if name in expected or Path(name).name != name or not name.endswith(".zip"):
            raise ValueError(f"bad or duplicate manifest entry: {name}")
        expected[name] = (digest, int(byte_count))

    seen = set()
    with tarfile.open(bundle, "r:xz") as archive:
        for member in archive:
            name = member.name
            if name not in expected or name in seen or not member.isfile():
                raise ValueError(f"unexpected, duplicate, or non-file member: {name}")
            digest, byte_count = expected[name]
            if member.size != byte_count:
                raise ValueError(f"wrong tar member size: {name}")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch: {name}")
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                bad_member = zipped.testzip()
                if bad_member is not None:
                    raise ValueError(f"ZIP CRC failure: {name} / {bad_member}")
            seen.add(name)

    if seen != expected.keys():
        raise ValueError(f"missing archives: {sorted(expected.keys() - seen)}")
    return {
        "artifact_only": True,
        "native_h5m_validation": "NOT_RUN",
        "archive_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "zip_archives_verified": len(seen),
        "total_original_zip_bytes": sum(n for _, n in expected.values()),
        "status": "PASS",
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_recovery.py archive.tar.xz WAVE_ARCHIVES.sha256")
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
