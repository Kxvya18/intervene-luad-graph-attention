from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


FILES = [
    "Model.csv",
    "CRISPRGeneEffect.csv",
    "OmicsSomaticMutations.csv",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "OmicsCNGene.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lines = [
        "# Data Manifest",
        "",
        "Release: DepMap Public 24Q4, per user-provided download selection.",
        "Recorded by script from local files.",
        "",
        "| file | size_bytes | modified_time_local | sha256 |",
        "|---|---:|---|---|",
    ]
    for name in FILES:
        path = RAW / name
        if not path.exists():
            lines.append(f"| {name} | missing | missing | missing |")
            continue
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        lines.append(f"| {name} | {stat.st_size} | {modified} | {sha256(path)} |")

    manifest = RAW / "DATA_MANIFEST.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
