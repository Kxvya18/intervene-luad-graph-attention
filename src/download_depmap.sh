#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw/depmap"
mkdir -p "$RAW_DIR"

RELEASE="${1:-DepMap Public 26Q1}"

download_file() {
  local filename="$1"
  local encoded_release
  encoded_release="$(python3 - <<PY
from urllib.parse import quote
print(quote("$RELEASE"))
PY
)"
  local url="https://depmap.org/portal/download/all/?releasename=${encoded_release}&filename=${filename}"
  echo "Downloading ${filename} from ${RELEASE}"
  curl -L --fail --retry 3 --output "${RAW_DIR}/${filename}" "$url"
}

download_file "CRISPRGeneEffect.csv"
download_file "Model.csv"

echo "Done. Files saved under ${RAW_DIR}"
