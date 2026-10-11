#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/boolu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"

cd "$PROJECT_ROOT"

echo "[1/7] GraphSparse Relapse-INTERVENE V3"
"$PYTHON_BIN" src/run_relapse_intervene_graph_sparse_v3.py

echo "[2/7] Robustness tests: permutation, bootstrap, clinical adjustment, module removal"
"$PYTHON_BIN" src/run_final_intervene_robustness_tests.py

echo "[3/7] Refined GSE31210 mechanism axes"
"$PYTHON_BIN" src/run_gse31210_refined_mechanism_axes.py

echo "[4/7] MRPL21/INTERVENE AI-biology validation"
"$PYTHON_BIN" src/run_mrpl21_intervene_ai_validation.py

echo "[5/7] TCBB benchmark confidence intervals and Cox baselines"
"$PYTHON_BIN" src/run_tcbb_final_benchmarks_ci.py

echo "[6/7] Publication SVG figures"
"$PYTHON_BIN" src/make_tcbb_final_publication_figures.py

echo "[7/7] Q1 readiness dashboard and registries"
"$PYTHON_BIN" src/make_tcbb_q1_readiness_package.py

echo "Done. See results/tcbb_q1_evidence_dashboard.md and paper_notes/tcbb_reproducibility_manifest.md"
