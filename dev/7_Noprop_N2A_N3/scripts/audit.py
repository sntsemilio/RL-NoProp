import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension.audit import audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["N2A", "N3pilot", "N3"])
    args = parser.parse_args()
    print(audit(args.stage))
