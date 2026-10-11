from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
PRISM = ROOT / "data" / "raw" / "PRISM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted([p for p in PRISM.iterdir() if p.is_file()])
    lines = [
        "# PRISM Data Manifest",
        "",
        "Release: PRISM Repurposing Public 24Q2, per downloaded filenames.",
        "Recorded by script from local files.",
        "",
        "| file | size_bytes | modified_time_local | sha256 |",
        "|---|---:|---|---|",
    ]
    for path in files:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        lines.append(f"| {path.name} | {stat.st_size} | {modified} | {sha256(path)} |")

    manifest = PRISM / "PRISM_DATA_MANIFEST.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
