#!/usr/bin/env python3
"""Precommit a format-only canonical/local ID repair after the preserved detection preflight failure."""
from __future__ import annotations
import copy,json
from pathlib import Path
from common import CORE3,EXP,freeze_json,now,read_jsonl,sha_file,atomic_json
IA=EXP/"IA_STD_Q15_API1";POST=IA/"post_ready"
OLD=POST/"configs"/"IA_API1_DETECTION_PRECOMMIT.json"
NEW=POST/"configs"/"IA_API1_DETECTION_ID_REPAIR_PRECOMMIT.json"

def main():
    expected=OLD.with_suffix('.sha256').read_text().split()[0]
    if sha_file(OLD)!=expected:raise RuntimeError('original detection precommit drift')
    old=json.loads(OLD.read_text(encoding='utf-8'))
    queries=read_jsonl(Path(old['queries']['path']));doc_ids={r['document_id'] for r in read_jsonl(Path(old['frozen_corpus']['path']))}
    member={r['canonical_target_id'] for r in queries if r['_membership']=='member'};nonmember={r['canonical_target_id'] for r in queries if r['_membership']=='nonmember'}
    local={r['target_doc_id'] for r in queries};canonical={r['canonical_target_id'] for r in queries}
    audit={'created_utc':now(),'failure_class':'ID_NAMESPACE_MISMATCH_BEFORE_PERFORMANCE','performance_observed':False,
      'queries':len(queries),'sessions':len({r['session_id'] for r in queries}),'member_sessions':len(member),'nonmember_sessions':len(nonmember),
      'local_ids_in_canonical_corpus':sum(v in doc_ids for v in local),'member_canonical_ids_in_corpus':sum(v in doc_ids for v in member),
      'nonmember_canonical_ids_in_corpus':sum(v in doc_ids for v in nonmember),'expected_member_inside':len(member),'expected_nonmember_inside':0,
      'repair':'use canonical_target_id for retrieval provenance; require every valid member target inside and every valid nonmember target outside',
      'changed':['ID namespace mapping','membership-aware provenance assertion'],'unchanged':['cohort','queries','labels','retriever','MIRABEL','Final LC','threshold rule','FPR budgets','bootstrap','hard gate']}
    if audit['member_canonical_ids_in_corpus']!=len(member) or audit['nonmember_canonical_ids_in_corpus']!=0:raise RuntimeError('canonical membership provenance failed')
    atomic_json(POST/'audits'/'IA_API1_ID_NAMESPACE_FAILURE_AUDIT.json',audit)
    pre=copy.deepcopy(old);pre.update({'created_utc':now(),'repair_parent':{'path':str(OLD),'sha256':sha_file(OLD)},'id_mapping_repair':audit,
      'code':{'path':str(EXP/'code'/'run_phase_c_detection_id_repair.py'),'sha256':sha_file(EXP/'code'/'run_phase_c_detection_id_repair.py')},
      'legacy_computation_module':{'path':str(EXP/'code'/'ia_detection_computation_id_repair.py'),'sha256':sha_file(EXP/'code'/'ia_detection_computation_id_repair.py')}})
    digest=freeze_json(NEW,pre)
    print(json.dumps({'precommit':str(NEW),'sha256':digest,'audit':audit},indent=2))
if __name__=='__main__':main()
