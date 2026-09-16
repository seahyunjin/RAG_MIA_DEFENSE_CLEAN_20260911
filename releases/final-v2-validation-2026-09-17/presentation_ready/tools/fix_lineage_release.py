#!/usr/bin/env python3
"""CPU-only lineage correction for the Final V2 release.

No model loading, embedding inference, retrieval, answer generation, threshold
rule changes, or attack-conditioned tuning are performed. Existing 1cba/dd1c
embedding caches are projected onto the frozen L1 corpus only.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, tempfile
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[4]
REL=ROOT/"releases/final-v2-validation-2026-09-17"
EXP=ROOT/"experiments"
TABLES=REL/"tables"; AUDITS=REL/"audits"; QUAR=REL/"quarantine"
L1_DB=EXP/"CLEAN_CORE3_DEV_V1/inputs/CLEAN_CORE3_PROTECTED_DB.jsonl"
L1_DOC_EMB=EXP/"CLEAN_CORE3_DEV_V1/cache/CORPUS_EMBEDDINGS.float16.npy"
AB_QUERY_EMB=EXP/"FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1/cache/GOLD_RECALIBRATION_QUERY_EMBEDDINGS.float16.npy"
A_IDS=EXP/"FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1/inputs/TOPIOCQA_BENIGN_RECALIBRATION_500.jsonl"
B_IDS=EXP/"FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/TOPIOCQA_GOLD_EVAL_1000.jsonl"
L2_RETRIEVAL=EXP/"FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/cache/GOLD_RETRIEVAL_AND_DETECTION.jsonl"
L2_DB=EXP/"FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/TOPIOCQA_GOLD_CORPUS.jsonl"
D_DB=EXP/"FINAL_V2_FINQA_ABLATION_ADAPTIVE_V1/inputs/FINQA_PROTECTED_DB.jsonl"
D_SCORES=EXP/"FINAL_V2_FINQA_ABLATION_ADAPTIVE_V1/runtime/FINQA_RETRIEVAL_AND_DETECTION.jsonl"
D_MATCHED=EXP/"NONMEMBER_INTERVENTION_COST_CPU_V1/tables/MATCHED_INTERVENTION_EFFICIENCY.csv"
E_SCORES=EXP/"FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/runtime/HARD_BENIGN_RETRIEVAL.jsonl"
E_SPLIT=EXP/"HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/HARD_BENIGN_SPLIT.csv"
CORE4=EXP/"FINAL_8ATTACK_E2E_FRAMEWORK_V1/cache/FINAL_RETRIEVAL_AND_DETECTION.jsonl"
DCMI=EXP/"FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/cache/CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl"
FROZEN={"G4":0.0901547372341156,"M":0.09579957418751517}
SEED=20260916

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p:Path,s:str):
 p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def write_json(p,obj):atomic(p,json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def read_jsonl(p):
 with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def read_csv(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 rows=list(rows);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def strict_tau(x,alpha=.025):
 x=np.sort(np.asarray(x,float));allow=math.floor(alpha*len(x)+1e-12)
 return float(x[-(allow+1)]) if allow else float(np.nextafter(x[-1],np.inf))
def score(r,name):
 if name=='s1':return float(r.get('s1',r.get('top_scores',[0])[0]))
 if name=='G4':
  if 'G4' in r:return float(r['G4'])
  if 'G' in r:return float(r['G'])
  x=np.asarray(r['top_scores'][:4],float);return float(x[0]-x.mean())
 if name=='M':return float(r['M'])
 raise KeyError(name)
def auc(negative,positive):
 neg=np.sort(np.asarray(negative,float));pos=np.asarray(positive,float)
 left=np.searchsorted(neg,pos,'left');right=np.searchsorted(neg,pos,'right')
 return float(np.mean((left+.5*(right-left))/len(neg)))
def rate(rows,name,tau):return float(np.mean([score(r,name)>tau for r in rows])) if rows else None
def stable(identity):return hashlib.sha256(f'{SEED}|MIXED_HARD500|{identity}'.encode()).hexdigest()

def project_ab_l1():
 docs=read_jsonl(L1_DB);doc_ids=[r['document_id'] for r in docs]
 d=np.asarray(np.load(L1_DOC_EMB,mmap_mode='r'),dtype=np.float32)
 q=np.asarray(np.load(AB_QUERY_EMB,mmap_mode='r'),dtype=np.float32)
 a_ids=[r['query_id'] for r in read_jsonl(A_IDS)];b_ids=[r['query_id'] for r in read_jsonl(B_IDS)]
 if d.shape!=(3000,1024) or q.shape!=(1500,1024) or len(a_ids)!=500 or len(b_ids)!=1000:raise RuntimeError('L1 shape/count drift')
 pre={"status":"PRECOMMITTED_BEFORE_SCORE_SAVE","cpu_only":True,"new_embedding_inference":0,"formula":{"G4":"s1-mean(s1..s4)","M":"canonical_official_v2"},"inputs":{str(x.relative_to(ROOT)):sha(x) for x in (L1_DB,L1_DOC_EMB,AB_QUERY_EMB,A_IDS,B_IDS)},"expected_order":"first 500=A, last 1000=B"}
 write_json(AUDITS/'LINEAGE_RECALC_PRECOMMIT.json',pre)
 n=d.shape[0];root=math.sqrt(2*math.log(n));g=-math.log(-math.log(.95));out=[]
 ids=a_ids+b_ids
 for st in range(0,len(q),128):
  mat=q[st:st+128]@d.T
  idx=np.argpartition(-mat,3,axis=1)[:,:4]
  vals=np.take_along_axis(mat,idx,axis=1);order=np.argsort(-vals,axis=1,kind='stable');idx=np.take_along_axis(idx,order,axis=1);vals=np.take_along_axis(vals,order,axis=1)
  s1=vals[:,0].astype(np.float64);g4=s1-vals.mean(axis=1,dtype=np.float64)
  sums=mat.sum(axis=1,dtype=np.float64);sqs=np.square(mat,dtype=np.float64).sum(axis=1,dtype=np.float64)
  mean=(sums-s1)/(n-1);var=np.maximum(0,(sqs-s1*s1)/(n-1)-mean*mean);std=np.sqrt(var);m=s1-(mean+std*root+g*std/root)
  for j in range(len(vals)):
   i=st+j;out.append({"query_id":ids[i],"benign_set":"A" if i<500 else "B","s1":float(s1[j]),"G4":float(g4[j]),"M":float(m[j]),"top_document_ids":"|".join(doc_ids[int(x)] for x in idx[j]),"top_scores":"|".join(f'{float(x):.10g}' for x in vals[j]),"retrieval_db":"L1","retrieval_db_hash":sha(L1_DB),"query_embedding_hash":sha(AB_QUERY_EMB),"document_embedding_hash":sha(L1_DOC_EMB)})
 a=out[:500];b=out[500:];med=float(np.median([r['s1'] for r in b]))
 if not .42<med<.45:raise RuntimeError(f'L1 validation failed: B s1 median={med}')
 write_csv(AUDITS/'TOPI_AB_ON_L1_SCORES.csv',out)
 write_json(AUDITS/'TOPI_AB_ON_L1_AUDIT.json',{"verdict":"PASS","A_n":500,"B_n":1000,"A_s1_median":float(np.median([r['s1'] for r in a])),"B_s1_median":med,"saved_score_sha256":sha(AUDITS/'TOPI_AB_ON_L1_SCORES.csv')})
 return a,b

def load_core():
 r=read_jsonl(CORE4);valid=[]
 for x in r:
  if x.get('attack') in {'MEntA','MBA','RAG-MIA'}:valid.append(x)
  elif x.get('attack')=='S²-MIA' and x.get('evaluation_split')=='S2_EVALUATION':valid.append(x)
 valid+=read_jsonl(DCMI)
 mem=[x for x in valid if x.get('membership')=='member'];non=[x for x in valid if x.get('membership')=='nonmember']
 if len(mem)!=9799 or len(non)!=9799:raise RuntimeError('Core5 count drift')
 return mem,non

def load_hard():
 rows={r['query_id']:r for r in read_jsonl(E_SCORES)};split={r['query_id']:r['split'] for r in read_csv(E_SPLIT)}
 if len(rows)!=2113 or set(rows)!=set(split):raise RuntimeError('D-hard split drift')
 cal=sorted([rows[q] for q in rows if split[q]=='CALIBRATION'],key=lambda r:r['query_id']);lock=sorted([rows[q] for q in rows if split[q]=='LOCKED_TEST'],key=lambda r:r['query_id'])
 hard500=sorted(cal,key=lambda r:(stable(r['query_id']),r['query_id']))[:500]
 return cal,lock,hard500

def quarantine():
 reason='Core member(L1) 점수와 TopiOCQA DB(L2) 점수를 조인함'
 groups={
  '01_signal_calibration_grid_l1xl2':[EXP/'H1_H2_SIGNAL_CALIBRATION_DIAG_CPU_V1/tables/TABLE_SIGNAL_CALIBRATION_GRID.csv'],
  '02_auc_06795_l1xl2':[EXP/'FINAL_V2_LINEAGE_FINQA_AUC_AUDIT_CPU_V1/tables/TABLE_THRESHOLD_FREE_AUC.csv'],
  '03_frozen_threshold_on_gold_b_l1xl2':[EXP/'FINAL_V2_LINEAGE_FINQA_AUC_AUDIT_CPU_V1/tables/TABLE_LINEAGE_RECONCILIATION.csv',EXP/'FINAL_V2_LINEAGE_FINQA_AUC_AUDIT_CPU_V1/tables/TABLE_FINQA_THRESHOLD_TRANSFER.csv'],
  '04_core_topiocqa_s1_0708_l1xl2':[EXP/'FINQA_CORE_GEOMETRY_DIAG_CPU_V1/tables/DISTRIBUTION_SUMMARY.csv']}
 for name,paths in groups.items():
  dst=QUAR/name;dst.mkdir(parents=True,exist_ok=True)
  copied=[]
  for src in paths:
   if src.is_file():shutil.copy2(src,dst/src.name);copied.append({"file":src.name,"source":str(src.relative_to(ROOT)),"sha256":sha(src)})
  atomic(dst/'INVALID_L1xL2_MIX.md',f'# INVALID L1×L2 MIX\n\n- 판정: `INVALID_L1xL2_MIX`\n- 원인: {reason}\n- 이 디렉터리의 수치는 발표·논문·모델 선택에 사용하지 않는다.\n')
  write_json(dst/'PROVENANCE.json',{"verdict":"INVALID_L1xL2_MIX","reason":reason,"files":copied})

def main():
 for p in (TABLES,AUDITS,QUAR):p.mkdir(parents=True,exist_ok=True)
 # Reclassify the prior hard-benign audit rather than erasing its failed premise.
 hard_ids={r['document_id'] for r in read_jsonl(L1_DB)};hard=read_jsonl(E_SCORES);all_ids=[x for r in hard for x in r['top_document_ids']]
 write_json(AUDITS/'HARD_BENIGN_DB_LINEAGE.json',{"previous_verdict":"HARD_BENIGN_DB_LINEAGE_MISMATCH_STOP","verdict":"RECLASSIFIED_D_HARD_FINQA","query_count":len(hard),"top_document_id_occurrences":len(all_ids),"in_L1":sum(x in hard_ids for x in all_ids),"finqa_prefix":sum(x.startswith('FinQA::') for x in all_ids),"hard_benign_retrieval_sha256":sha(E_SCORES),"retrieval_db":"D-hard(E)","retrieval_db_hash":sha(D_DB),"note":"D-hard uses the FinQA protected DB; it is not Core/L1."})
 a,b=project_ab_l1();core_m,core_n=load_core();hcal,hlock,h500=load_hard();fin=read_jsonl(D_SCORES);fmem=[r for r in fin if r.get('membership')=='member'];fnon=[r for r in fin if r.get('membership')=='nonmember'];dcal=[r for r in fin if r.get('cohort')=='BENIGN_CALIBRATION'];dlock=[r for r in fin if r.get('cohort')=='BENIGN_LOCKED_TEST']
 if len(fmem)!=9995 or len(fnon)!=9995 or len(dcal)!=1000 or len(dlock)!=1000:raise RuntimeError('FinQA count drift')
 # 3a
 frozen=[]
 for name in ('G4','M'):
  tau=FROZEN[name];frozen.append({"score":name,"threshold":tau,"comparator":">","benign_set":"B","benign_n":len(b),"alarm_n":sum(score(r,name)>tau for r in b),"actual_fpr":rate(b,name,tau),"retrieval_db":"L1","retrieval_db_hash":sha(L1_DB),"embedding_hash":sha(AB_QUERY_EMB)})
 write_csv(TABLES/'FROZEN_THRESHOLD_ON_TOPI_L1.csv',frozen)
 # 3b
 auc_rows=[]
 for name in ('s1','G4','M'):
  val=auc([score(r,name) for r in b],[score(r,name) for r in core_m]);auc_rows.append({"score":name,"negative_set":"B-on-Core","negative_n":len(b),"positive_set":"Core5 member","positive_n":len(core_m),"auc":val,"effective_auc":max(val,1-val),"retrieval_db":"L1","retrieval_db_hash":sha(L1_DB),"embedding_hash":sha(AB_QUERY_EMB)})
 write_csv(TABLES/'AUC_CORE_VS_TOPI_L1.csv',auc_rows)
 # 3c/3d
 banks={'A_ON_L1':a,'E_CAL_D_HARD':hcal,'MIXED_A_L1_HARD500':a+h500};grid=[]
 for name in ('s1','G4','M'):
  for bank,rows in banks.items():
   tau=strict_tau([score(r,name) for r in rows]);row={"score":name,"calibration_bank":bank,"threshold":tau,"comparator":">","calibration_n":len(rows),"calibration_actual_fpr":rate(rows,name,tau),"B_on_L1_n":len(b),"B_on_L1_fpr":rate(b,name,tau),"E_lock_n":len(hlock),"E_lock_fpr":rate(hlock,name,tau)}
   if bank=='E_CAL_D_HARD':
    row.update({"matched_attack_lineage":"FinQA/D","FinQA_member_n":len(fmem),"FinQA_member_tpr":rate(fmem,name,tau),"FinQA_nonmember_n":len(fnon),"FinQA_nonmember_intervention":rate(fnon,name,tau),"Core5_member_tpr":"","Core5_nonmember_intervention":"","comparison_type":"FinQA 동일 코퍼스 매칭 비교"})
   else:
    row.update({"matched_attack_lineage":"Core5/L1" if bank=='A_ON_L1' else 'mixed calibration -> Core5/L1',"Core5_member_n":len(core_m),"Core5_member_tpr":rate(core_m,name,tau),"Core5_nonmember_n":len(core_n),"Core5_nonmember_intervention":rate(core_n,name,tau),"FinQA_member_tpr":"","FinQA_nonmember_intervention":"","comparison_type":"L1 matched" if bank=='A_ON_L1' else 'cross-corpus mixed calibration diagnostic'})
   row.update({"benign_set":'A' if bank=='A_ON_L1' else ('E-cal' if bank=='E_CAL_D_HARD' else 'A+E500'),"benign_set_hash":sha(A_IDS) if bank=='A_ON_L1' else (sha(E_SPLIT) if bank=='E_CAL_D_HARD' else hashlib.sha256((sha(A_IDS)+sha(E_SPLIT)).encode()).hexdigest()),"retrieval_db":'L1' if bank=='A_ON_L1' else ('D-hard(E)' if bank=='E_CAL_D_HARD' else 'L1+D-hard mixed'),"retrieval_db_hash":sha(L1_DB) if bank=='A_ON_L1' else (sha(D_DB) if bank=='E_CAL_D_HARD' else hashlib.sha256((sha(L1_DB)+sha(D_DB)).encode()).hexdigest()),"embedding_hash":sha(AB_QUERY_EMB) if bank=='A_ON_L1' else 'MISSING_SAVED_EMBEDDING_CACHE'})
   grid.append(row)
 write_csv(TABLES/'SIGNAL_CALIBRATION_GRID_L1.csv',grid);write_csv(TABLES/'LINEAGE_CORRECTED_CALIBRATION_SUMMARY.csv',grid)
 # Exact-D-lock FPR and D-hard percentile rows, side-by-side within FinQA.
 matched={r['method']:r for r in read_csv(D_MATCHED) if r['setting']=='FINQA' and r['attack']=='POOLED_ATTACK_QUERIES'}
 fin_rows=[]
 for name,label,mkey in [('G4','Final V2','G4'),('M','MIRABEL','MIRABEL')]:
  te=strict_tau([score(r,name) for r in hcal]); mr=matched[mkey]
  fin_rows.append({"method":label,"calibration":"D-cal / D-lock exact-FPR diagnostic","estimator":"locked benign exact FPR 2.5%","comparison_label":"FinQA 동일 코퍼스 매칭 비교","threshold":mr['threshold'],"member_n":mr['member_n'],"member_tpr":mr['member_tpr'],"nonmember_n":mr['nonmember_n'],"nonmember_intervention":mr['nonmember_tpr'],"matched_locked_set":"D-lock","matched_locked_fpr":mr['actual_benign_fpr'],"E_lock_reference_fpr":rate(hlock,name,te),"benign_set":"D-lock","benign_set_hash":sha(D_SCORES),"retrieval_db":"D","retrieval_db_hash":sha(D_DB),"embedding_hash":"MISSING_SAVED_EMBEDDING_CACHE","source_table":str(D_MATCHED.relative_to(ROOT))})
  fin_rows.append({"method":label,"calibration":"E-cal (D-hard: FinQA hard benign)","estimator":"E-cal empirical 97.5 percentile","comparison_label":"FinQA 동일 코퍼스 매칭 비교","threshold":te,"member_n":len(fmem),"member_tpr":rate(fmem,name,te),"nonmember_n":len(fnon),"nonmember_intervention":rate(fnon,name,te),"matched_locked_set":"E-lock","matched_locked_fpr":rate(hlock,name,te),"E_lock_reference_fpr":rate(hlock,name,te),"benign_set":"E-cal","benign_set_hash":sha(E_SPLIT),"retrieval_db":"D-hard(E)","retrieval_db_hash":sha(D_DB),"embedding_hash":"MISSING_SAVED_EMBEDDING_CACHE","source_table":"experiments/HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_THRESHOLDS_AND_LOCKED_FPR.csv"})
 write_csv(TABLES/'FINQA_HARD_BENIGN_MATCHED.csv',fin_rows)
 # Cross-corpus E threshold -> Core5. Not matched.
 cross=[]
 for name,label in [('G4','Final V2'),('M','MIRABEL')]:
  tau=strict_tau([score(r,name) for r in hcal]);ma=sum(score(r,name)>tau for r in core_m);na=sum(score(r,name)>tau for r in core_n);ba=sum(score(r,name)>tau for r in hlock)
  cross.append({"method":label,"label":"FinQA hard benign 임계를 BEIR Core5 공격에 적용","comparison_type":"cross-corpus threshold transfer; NOT matched","threshold":tau,"E_lock_n":len(hlock),"E_lock_alarm_n":ba,"E_lock_fpr":ba/len(hlock),"Core5_member_n":len(core_m),"Core5_member_alarm_n":ma,"Core5_member_tpr":ma/len(core_m),"Core5_nonmember_n":len(core_n),"Core5_nonmember_alarm_n":na,"Core5_nonmember_intervention":na/len(core_n),"precision":ma/(ma+na+ba),"benign_set":"E-cal/E-lock","benign_set_hash":sha(E_SPLIT),"retrieval_db":"D-hard(E) -> L1","retrieval_db_hash":f'{sha(D_DB)} -> {sha(L1_DB)}',"embedding_hash":"MISSING_CROSS_CORPUS_MULTIPLE"})
 write_csv(TABLES/'CROSS_CORPUS_THRESHOLD_TRANSFER.csv',cross)
 type_src=read_csv(EXP/'HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_HARD_BENIGN_TYPE_FPR.csv');types=[]
 for r in type_src:
  if r['estimator']=='EMPIRICAL_P97_5':types.append({**r,"label":"FinQA D-hard(E) locked benign type FPR","comparison_type":"D-hard within-corpus FPR; Core5 matching claim prohibited","benign_set":"E-lock","benign_set_hash":sha(E_SPLIT),"retrieval_db":"D-hard(E)","retrieval_db_hash":sha(D_DB),"embedding_hash":"MISSING_SAVED_EMBEDDING_CACHE"})
 write_csv(TABLES/'CROSS_CORPUS_THRESHOLD_TRANSFER_BY_HARD_TYPE.csv',types)
 quarantine()
 lineage={"L1":{"description":"BEIR Core DB 3,000","retrieval_db_path":str(L1_DB.relative_to(ROOT)),"retrieval_db_hash":sha(L1_DB),"document_embedding_hash":sha(L1_DOC_EMB),"benign_sets":["A","B","C","C-Sci"]},"L2":{"description":"TopiOCQA utility DB only","retrieval_db_path":str(L2_DB.relative_to(ROOT)),"retrieval_db_hash":sha(L2_DB),"retrieval_artifact_hash":sha(L2_RETRIEVAL),"benign_sets":["TopiOCQA Gold utility"]},"D":{"description":"FinQA attack/benign DB","retrieval_db_path":str(D_DB.relative_to(ROOT)),"retrieval_db_hash":sha(D_DB),"benign_sets":["D-cal","D-lock"]},"D-hard(E)":{"description":"FinQA hard-benign bank; not Core","retrieval_db_path":str(D_DB.relative_to(ROOT)),"retrieval_db_hash":sha(D_DB),"hard_retrieval_hash":sha(E_SCORES),"split_hash":sha(E_SPLIT),"benign_sets":["E-cal","E-lock","E-full"]}}
 write_json(AUDITS/'RETRIEVAL_LINEAGE.json',lineage)
 write_json(AUDITS/'LINEAGE_CORRECTION_RESULT.json',{"verdict":"LINEAGE_CORRECTION_PASS","B_on_L1_s1_median":float(np.median([r['s1'] for r in b])),"frozen_threshold_B_on_L1":frozen,"quarantine_groups":4,"outputs":[p.name for p in sorted(TABLES.glob('*.csv'))]})
 print(json.dumps({"verdict":"LINEAGE_CORRECTION_PASS","B_s1_median":float(np.median([r['s1'] for r in b])),"frozen_B_FPR":{r['score']:r['actual_fpr'] for r in frozen},"tables":len(list(TABLES.glob('*.csv')))},indent=2))
if __name__=='__main__':main()
