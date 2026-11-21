#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
python -m pytest tests/test_gymnasium_conformance.py tests/test_environments.py tests/test_environment_solvability.py
AURA_REQUIRE_GYMNASIUM=1 python -m pytest tests/test_gymnasium_conformance.py
