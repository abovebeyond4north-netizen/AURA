#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -ra > verification/v110/full_suite.log 2>&1
rc=$?; echo "$rc" > verification/v110/full_suite.exit
exit "$rc"
