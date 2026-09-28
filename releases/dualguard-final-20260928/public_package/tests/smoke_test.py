import json,torch
from dualguard import cld4,apply_cld,cf_clip_logits
assert abs(cld4([.8,.6,.59,.58])-.1575)<1e-12
r=apply_cld([.8,.6,.59,.58],["a","b","c","d"]);assert r["alarm"] and r["removed"]=="a"
full=torch.tensor([1.,2.,3.]);fl=torch.log_softmax(full,0);drop=torch.stack([fl-torch.tensor([0.,10.,0.]),fl]);out,lift,clip=cf_clip_logits(full,fl,drop);assert out.shape==full.shape and torch.all(clip>=0)
print(json.dumps({"status":"SMOKE_PASS","g4":r["g4"],"alarm":r["alarm"]}))
