#!/usr/bin/env python3
"""Run budgeted agent learning without granting shell/network access to learner."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts.train import main
if __name__=='__main__':main()
