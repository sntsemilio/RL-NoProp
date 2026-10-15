"""Complementary campaign; historical implementation is imported read-only."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "6_Noprop_Neural"
sys.path.insert(0, str(OLD))
