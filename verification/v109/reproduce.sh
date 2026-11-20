#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -ra
AURA_TEST_NAMESPACES=1 python -m pytest -o addopts= -q tests/test_namespace_probe.py
