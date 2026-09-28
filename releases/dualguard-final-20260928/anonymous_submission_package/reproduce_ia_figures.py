import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/TABLE_SESSION_ATTACKS_Q1_Q5.csv";r=[x for x in csv.DictReader(f.open()) if x["Attack"]=="IA"];assert r and all(f"Q{i}" in r[0] for i in range(1,6));print("DRY_RUN_PASS" if a.dry_run else "IA trajectory schema valid")
