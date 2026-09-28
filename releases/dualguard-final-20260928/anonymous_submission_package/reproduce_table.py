import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/FINAL_ATTACK_PERFORMANCE_TABLE.csv";r=list(csv.DictReader(f.open()));assert r
print("DRY_RUN_PASS" if a.dry_run else "\n".join(f"{x['attack']}: {x['method']}={x['AUROC_or_leakage']}" for x in r))
