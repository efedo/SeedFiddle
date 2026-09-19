"""Batch the saved project recipe using the desktop result contract."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from seedvision.export.project_runner import run_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True,help='Saved, fingerprint-verified project master')
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'project-results')
    args = parser.parse_args()
    for path in run_project(args.workspace,args.project,args.output_dir):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
