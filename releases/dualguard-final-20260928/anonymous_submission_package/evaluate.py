#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--csv",type=Path);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
 if a.dry_run: print(json.dumps({"status":"DRY_RUN_PASS","expected_columns":["attack","method","AUROC_or_leakage"]}));return
 rows=list(csv.DictReader(a.csv.open()));print(json.dumps({"rows":len(rows)},indent=2))
if __name__=="__main__":main()
