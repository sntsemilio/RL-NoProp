import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension.analysis import analyze_n2a, analyze_n3, report
from extension.io import manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["N2A", "N3", "historical", "report"])
    args = parser.parse_args()
    if args.stage in ("N2A", "historical"):
        print(analyze_n2a(require_complete=args.stage == "N2A"))
    elif args.stage == "N3":
        analyze_n3()
    else:
        report()
    print("Manifest", manifest())
