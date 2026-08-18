#!/usr/bin/env python3
"""Compatibility wrapper. Use mtp_gate5_replay.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('mtp_gate5_replay.py')), run_name='__main__')
