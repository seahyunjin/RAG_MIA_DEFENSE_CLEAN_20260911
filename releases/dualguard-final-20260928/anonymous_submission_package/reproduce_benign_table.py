import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/FINAL_CORRECTED_BENIGN_UTILITY.csv";r=list(csv.DictReader(f.open()));assert r;print("DRY_RUN_PASS" if a.dry_run else len(r))
