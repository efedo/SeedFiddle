"""Batch the saved project recipe using the desktop result contract."""
from pathlib import Path
import argparse
import json
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
    parser.add_argument('--seed-crop-padding-px',type=int,default=None,
        help='Also export each proposed seed image/mask crop with this nonnegative pixel padding')
    args = parser.parse_args()
    if args.seed_crop_padding_px is not None and args.seed_crop_padding_px < 0:
        parser.error('--seed-crop-padding-px must be nonnegative')
    for path in run_project(args.workspace,args.project,args.output_dir,crop_padding_px=args.seed_crop_padding_px):
        print(path)
        if args.seed_crop_padding_px is not None:
            manifests = list(path.glob('seed-crops-*/seed-crops.json'))
            if not manifests:
                print(f'No seed crops are available for {path}; see its result report.', file=sys.stderr)
            for manifest in manifests:
                records = json.loads(manifest.read_text(encoding='utf-8'))['records']
                incomplete = sum(record['status'] != 'available' for record in records)
                if incomplete:
                    print(f'{len(records) - incomplete} crops exported; {incomplete} requested seeds have no crop. '
                          f'See {manifest} for individual reasons.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
