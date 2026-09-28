#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from dualguard import cld4,apply_cld

def main():
 p=argparse.ArgumentParser();p.add_argument("--toy",action="store_true");p.add_argument("--input",type=Path);a=p.parse_args()
 if a.toy:
  row={"top_scores":[0.8,0.6,0.59,0.58],"documents":["d1","d2","d3","d4"]};print(json.dumps(apply_cld(row["top_scores"],row["documents"]),indent=2));return
 if not a.input: raise SystemExit("Provide --toy or --input. Full generation integration supplies FULL and DROP prompts to SerialKVCacheCFClip.")
 row=json.loads(a.input.read_text());print(json.dumps(apply_cld(row["top_scores"],row["documents"]),indent=2))
if __name__=="__main__":main()
