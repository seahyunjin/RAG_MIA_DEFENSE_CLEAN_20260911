#!/usr/bin/env python3
"""Build the new shared Core3 membership substrate without outcome signals."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
INPUTS = CAMPAIGN / "inputs"
AUDITS = CAMPAIGN / "audits"
MANIFESTS = CAMPAIGN / "manifests"

DOMAINS = ("nfcorpus", "scidocs", "trec-covid")
TARGET_QUOTAS = {"nfcorpus": 14, "scidocs": 14, "trec-covid": 12}
BENIGN_PER_SPLIT = {"nfcorpus": 250, "scidocs": 225, "trec-covid": 25}
DOCUMENTS_PER_DOMAIN = 1000
MASK_COUNT = 5

# Static lexical exclusions are input-validity checks only.  They are never
# changed after observing an attack, retrieval, detector, or answer outcome.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "may", "might", "more", "most", "no", "not", "of",
    "on", "or", "our", "ours", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def source_text(row: dict) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    return "\n".join(part for part in (title, text) if part)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def global_id(domain: str, local_id: str) -> str:
    return f"BeIR_{domain}::{local_id}"


def structural_eligibility(text: str) -> tuple[bool, dict]:
    words = WORD_RE.findall(text)
    factual_units = [x.strip() for x in re.split(r"[.!?;]+", text) if len(WORD_RE.findall(x)) >= 5]
    group_counts = []
    for i in range(MASK_COUNT):
        start = math.floor(i * len(words) / MASK_COUNT)
        end = math.floor((i + 1) * len(words) / MASK_COUNT)
        eligible = [w for w in words[start:end] if w.lower() not in STOPWORDS and len(w) > 1]
        group_counts.append(len(eligible))
    reasons = []
    if len(words) < 80:
        reasons.append("LT_80_WORDS")
    if len(factual_units) < 5:
        reasons.append("LT_5_FACTUAL_UNITS")
    if len(group_counts) != MASK_COUNT or min(group_counts, default=0) < 2:
        reasons.append("MBA_GROUP_WITH_LT_2_LEXICAL_CANDIDATES")
    return not reasons, {
        "word_count": len(words),
        "factual_unit_count": len(factual_units),
        "mba_group_eligible_counts": group_counts,
        "reasons": reasons,
    }


def load_qrels(domain: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in sorted((RAW / domain / "qrels").glob("*.tsv")):
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or row[0].lower() in {"query-id", "query_id"}:
                    continue
                if len(row) < 3:
                    continue
                qid, docid, score = row[0], row[1], row[2]
                try:
                    positive = float(score) > 0
                except ValueError:
                    continue
                if positive:
                    result.setdefault(qid, set()).add(docid)
    return result


def main() -> None:
    for directory in (INPUTS, AUDITS, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    docs_by_domain: dict[str, list[dict]] = {}
    docs_by_gid: dict[str, dict] = {}
    norm_hash_counts: dict[str, int] = {}
    eligible_by_domain: dict[str, list[dict]] = {}

    for domain in DOMAINS:
        docs = []
        eligible = []
        for raw_row in read_jsonl(RAW / domain / "corpus.jsonl"):
            local = str(raw_row.get("_id") or raw_row.get("id"))
            gid = global_id(domain, local)
            text = source_text(raw_row)
            norm = normalize_text(text)
            norm_hash = sha_text(norm)
            ok, details = structural_eligibility(text)
            row = {
                "document_id": gid,
                "local_document_id": local,
                "domain": domain,
                "title": str(raw_row.get("title") or ""),
                "text": str(raw_row.get("text") or ""),
                "source_text": text,
                "normalized_text_hash": norm_hash,
                "selection_key": sha_text(gid + "||" + norm_hash),
                "eligibility": details,
            }
            docs.append(row)
            docs_by_gid[gid] = row
            norm_hash_counts[norm_hash] = norm_hash_counts.get(norm_hash, 0) + 1
            if ok:
                eligible.append(row)
        docs_by_domain[domain] = docs
        eligible_by_domain[domain] = eligible

    # Duplicate-normalized documents are excluded from the common target pool.
    common_rows = []
    for domain in DOMAINS:
        for row in eligible_by_domain[domain]:
            if norm_hash_counts[row["normalized_text_hash"]] == 1:
                common_rows.append(row)

    pool_path = INPUTS / "COMMON_ELIGIBLE_POOL.csv.gz"
    with gzip.open(pool_path, "wt", encoding="utf-8", newline="") as f:
        fields = ["document_id", "local_document_id", "domain", "normalized_text_hash", "selection_key", "word_count", "factual_unit_count", "mba_group_eligible_counts"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(common_rows, key=lambda x: (x["domain"], x["selection_key"])):
            writer.writerow({
                "document_id": row["document_id"],
                "local_document_id": row["local_document_id"],
                "domain": row["domain"],
                "normalized_text_hash": row["normalized_text_hash"],
                "selection_key": row["selection_key"],
                "word_count": row["eligibility"]["word_count"],
                "factual_unit_count": row["eligibility"]["factual_unit_count"],
                "mba_group_eligible_counts": json.dumps(row["eligibility"]["mba_group_eligible_counts"]),
            })

    targets = []
    for domain in DOMAINS:
        eligible = sorted(
            [r for r in common_rows if r["domain"] == domain],
            key=lambda x: x["selection_key"],
        )
        quota = TARGET_QUOTAS[domain]
        if len(eligible) < quota:
            raise RuntimeError(f"insufficient common eligible targets for {domain}: {len(eligible)} < {quota}")
        for index, row in enumerate(eligible[:quota]):
            target = {k: row[k] for k in ("document_id", "local_document_id", "domain", "title", "text", "source_text", "normalized_text_hash", "selection_key")}
            target["membership"] = "member" if index % 2 == 0 else "nonmember"
            target["domain_selection_index"] = index
            targets.append(target)
    if sum(t["membership"] == "member" for t in targets) != 20 or sum(t["membership"] == "nonmember" for t in targets) != 20:
        raise RuntimeError("deterministic domain-balanced membership assignment did not produce 20/20")

    target_ids = {t["document_id"] for t in targets}
    target_norm_hashes = {t["normalized_text_hash"] for t in targets}
    with (INPUTS / "SHARED_TARGETS.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["document_id", "local_document_id", "domain", "membership", "normalized_text_hash", "selection_key", "domain_selection_index", "title", "source_text"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: t[k] for k in fields} for t in targets)

    # Benign split is selected before retrieval and excludes all attack targets.
    benign_cal = []
    benign_hold = []
    seen_query_hashes: set[str] = set()
    for domain in DOMAINS:
        query_rows = read_jsonl(RAW / domain / "queries.jsonl")
        qrels = load_qrels(domain)
        candidates = []
        for q in query_rows:
            local_qid = str(q.get("_id") or q.get("id"))
            text = str(q.get("text") or "").strip()
            nq = normalize_text(text)
            qhash = sha_text(nq)
            gold_local = sorted(qrels.get(local_qid, set()))
            gold_global = [global_id(domain, x) for x in gold_local if global_id(domain, x) in docs_by_gid]
            if not text or not gold_global or qhash in seen_query_hashes:
                continue
            if target_ids.intersection(gold_global):
                continue
            primary_gold = min(gold_global, key=sha_text)
            candidates.append({
                "query_id": f"BENIGN::{domain}::{local_qid}",
                "local_query_id": local_qid,
                "domain": domain,
                "query": text,
                "query_hash": qhash,
                "gold_document_ids": gold_global,
                "primary_gold_document_id": primary_gold,
                "selection_key": sha_text(f"BENIGN::{domain}::{local_qid}||{qhash}"),
            })
            seen_query_hashes.add(qhash)
        candidates.sort(key=lambda x: x["selection_key"])
        needed = BENIGN_PER_SPLIT[domain] * 2
        if len(candidates) < needed:
            raise RuntimeError(f"insufficient benign queries for {domain}: {len(candidates)} < {needed}")
        chosen = candidates[:needed]
        benign_cal.extend(chosen[: BENIGN_PER_SPLIT[domain]])
        benign_hold.extend(chosen[BENIGN_PER_SPLIT[domain] :])

    if len(benign_cal) != 500 or len(benign_hold) != 500:
        raise RuntimeError("benign split is not exactly 500/500")
    if {r["query_id"] for r in benign_cal} & {r["query_id"] for r in benign_hold}:
        raise RuntimeError("benign calibration/holdout overlap")
    write_jsonl(INPUTS / "BENIGN_CALIBRATION.jsonl", benign_cal)
    write_jsonl(INPUTS / "BENIGN_HOLDOUT.jsonl", benign_hold)

    # Reuse the previously frozen clean construction rule, not its prior DB.
    db_rows = []
    nonmember_ids = {t["document_id"] for t in targets if t["membership"] == "nonmember"}
    nonmember_hashes = {t["normalized_text_hash"] for t in targets if t["membership"] == "nonmember"}
    member_ids = {t["document_id"] for t in targets if t["membership"] == "member"}
    benign_gold_by_domain = {domain: set() for domain in DOMAINS}
    for row in benign_cal + benign_hold:
        benign_gold_by_domain[row["domain"]].add(row["primary_gold_document_id"])

    per_domain_counts = {}
    for domain in DOMAINS:
        mandatory_ids = (member_ids | benign_gold_by_domain[domain])
        mandatory = [docs_by_gid[x] for x in mandatory_ids if x in docs_by_gid and docs_by_gid[x]["domain"] == domain]
        candidates = [
            row for row in docs_by_domain[domain]
            if row["document_id"] not in nonmember_ids
            and row["normalized_text_hash"] not in nonmember_hashes
            and row["document_id"] not in {x["document_id"] for x in mandatory}
        ]
        candidates.sort(key=lambda x: sha_text(x["document_id"]))
        selected = mandatory + candidates[: max(0, DOCUMENTS_PER_DOMAIN - len(mandatory))]
        if len(selected) != DOCUMENTS_PER_DOMAIN:
            raise RuntimeError(f"could not construct {DOCUMENTS_PER_DOMAIN}-document DB for {domain}")
        selected.sort(key=lambda x: sha_text(x["document_id"]))
        db_rows.extend({
            "document_id": row["document_id"],
            "local_document_id": row["local_document_id"],
            "domain": row["domain"],
            "title": row["title"],
            "text": row["text"],
            "source_text": row["source_text"],
            "normalized_text_hash": row["normalized_text_hash"],
        } for row in selected)
        per_domain_counts[domain] = len(selected)

    db_path = INPUTS / "CLEAN_CORE3_PROTECTED_DB.jsonl"
    write_jsonl(db_path, db_rows)
    db_ids = {r["document_id"] for r in db_rows}
    db_hashes = {r["normalized_text_hash"] for r in db_rows}
    member_inclusion = sum(x in db_ids for x in member_ids)
    nonmember_id_leak = sum(x in db_ids for x in nonmember_ids)
    nonmember_text_leak = sum(x in db_hashes for x in nonmember_hashes)
    membership_pass = member_inclusion == 20 and nonmember_id_leak == 0 and nonmember_text_leak == 0 and not (member_ids & nonmember_ids)

    membership_audit = {
        "verdict": "CORE3_MEMBERSHIP_AUDIT_PASS" if membership_pass else "CORE3_MEMBERSHIP_AUDIT_FAILED",
        "member_targets": len(member_ids),
        "nonmember_targets": len(nonmember_ids),
        "member_inclusion": member_inclusion,
        "nonmember_id_in_db": nonmember_id_leak,
        "nonmember_normalized_text_duplicate_in_db": nonmember_text_leak,
        "member_nonmember_id_overlap": len(member_ids & nonmember_ids),
    }
    write_json(AUDITS / "CORE3_MEMBERSHIP_AUDIT.json", membership_audit)
    if not membership_pass:
        raise RuntimeError("membership audit failed")

    ordered_ids = [r["document_id"] for r in db_rows]
    ordered_hashes = [r["normalized_text_hash"] for r in db_rows]
    db_manifest = {
        "campaign": "CLEAN_CORE3_DEV_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "construction_rule": "per-domain 1000 documents; include frozen member targets and one SHA256-selected primary qrels gold source per benign query; exclude frozen nonmember IDs and normalized-text hashes; fill and order by SHA256(global document ID)",
        "documents": len(db_rows),
        "domain_counts": per_domain_counts,
        "ordered_document_id_sha256": sha_text("\n".join(ordered_ids)),
        "ordered_normalized_text_hash_sha256": sha_text("\n".join(ordered_hashes)),
        "corpus_file": str(db_path),
        "corpus_sha256": sha_file(db_path),
        "member_target_ids": sorted(member_ids),
        "nonmember_target_ids": sorted(nonmember_ids),
        "raw_sources": {domain: str(RAW / domain / "corpus.jsonl") for domain in DOMAINS},
    }
    write_json(MANIFESTS / "CLEAN_CORE3_DB_MANIFEST.json", db_manifest)

    pool_manifest = {
        "rule": "NFKC+lowercase+whitespace normalized unique text; >=80 lexical words; >=5 factual units; each of five equal word subtexts has >=2 non-stopword lexical candidates",
        "mask_count_precommitted": MASK_COUNT,
        "counts": {domain: sum(r["domain"] == domain for r in common_rows) for domain in DOMAINS},
        "total": len(common_rows),
        "pool_file": str(pool_path),
        "pool_sha256": sha_file(pool_path),
        "selection_rule": "domain quotas nfcorpus=14, scidocs=14, trec-covid=12; within-domain ascending SHA256(global_id || normalized_text_hash); alternating member/nonmember assignment",
        "targets_file_sha256": sha_file(INPUTS / "SHARED_TARGETS.csv"),
        "benign_calibration_sha256": sha_file(INPUTS / "BENIGN_CALIBRATION.jsonl"),
        "benign_holdout_sha256": sha_file(INPUTS / "BENIGN_HOLDOUT.jsonl"),
    }
    write_json(MANIFESTS / "COMMON_ELIGIBLE_POOL_MANIFEST.json", pool_manifest)

    result = {
        "verdict": "CORE3_MEMBERSHIP_AUDIT_PASS",
        "common_eligible_total": len(common_rows),
        "common_eligible_by_domain": pool_manifest["counts"],
        "targets": {"member": 20, "nonmember": 20, "domain_counts": TARGET_QUOTAS},
        "benign": {"calibration": 500, "holdout": 500, "per_split_domain_counts": BENIGN_PER_SPLIT},
        "database": {"documents": len(db_rows), "domain_counts": per_domain_counts, "sha256": db_manifest["corpus_sha256"]},
    }
    write_json(CAMPAIGN / "SUBSTRATE_RESULT.json", result)
    (CAMPAIGN / "STATUS.md").write_text(
        "# CLEAN_CORE3_DEV_V1 STATUS\n\n"
        "- Phase 1: `PHASE1_PROTOCOL_ARTIFACT_VERIFICATION_PASS`\n"
        "- Phase 2: `COMMON_ELIGIBLE_POOL_FROZEN`\n"
        "- Phase 3: `SHARED_20_20_TARGETS_FROZEN`\n"
        "- Phase 4: `PROTECTED_DB_FROZEN`\n"
        "- Phase 5: `CORE3_MEMBERSHIP_AUDIT_PASS`\n"
        "- Phase 6+: `NOT_STARTED`\n"
        "- Historical numeric results used: `NO`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
