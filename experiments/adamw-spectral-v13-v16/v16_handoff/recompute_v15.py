"""Read-only independent aggregation of the user-provided V15 light bundle.
Does not load checkpoints, contact a network, or run GPU training.
"""
from pathlib import Path
import json, hashlib, numpy as np
ROOT=Path('/mnt/data/v15_review_work')
OUT=Path('/mnt/data/v15_review_v16_handoff/evidence')
OUT.mkdir(parents=True,exist_ok=True)
def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def ev(branch,h=64,split='confirm'):
    return rows(ROOT/'raw/eval'/f'{branch}__H{h}__{split}.jsonl')
def nll(a):return sum(x['loss_sum'] for x in a)/sum(x['token_count'] for x in a)
def paired(a,b):
    assert [r['doc_id'] for r in a]==[r['doc_id'] for r in b]
    assert [r['token_count'] for r in a]==[r['token_count'] for r in b]
    d=np.array([x['loss_sum']-y['loss_sum'] for x,y in zip(a,b)],float)
    t=np.array([x['token_count'] for x in a],float)
    rng=np.random.default_rng(340715)
    ii=rng.integers(0,len(a),(5000,len(a)))
    bb=d[ii].sum(1)/t[ii].sum(1)
    return {'point':float(d.sum()/t.sum()),'ci95':np.quantile(bb,[.025,.975]).tolist(),
            'ci97p5':np.quantile(bb,[.0125,.9875]).tolist(), 'docs':len(a),'tokens':int(t.sum())}
branches=['S_a2_b1','F_a2_b1','P_a2_b1','S_a1_b1']
o={'method':'independent read-only CPU aggregation, no GPU/checkpoint validation',
   'nll':{b:{str(h):nll(ev(b,h)) for h in [1,64]} for b in branches},
   'main':{f:paired(ev('S_a2_b1'),ev(f'{f}_a2_b1')) for f in ['F','P']},
   'F_vs_P':paired(ev('P_a2_b1'),ev('F_a2_b1')),
   'one_shot':{f:{'S_minus_oneshot':paired(ev('S_a2_b1'),ev(f'{f}_a2_b1_oneshot')),
                  'oneshot_minus_sustained':paired(ev(f'{f}_a2_b1_oneshot'),ev(f'{f}_a2_b1'))}
                  for f in ['F','P']},
   'aux':{f:paired(ev('aux_S_a2_b1'),ev(f'aux_{f}_a2_b1')) for f in ['F','P']}}
# Show same-norm within a branch is not shared norm across evolving branches.
s=rows(ROOT/'raw/steps/S_a2_b1.jsonl')
o['norm_trajectory']={}
for f in ['F','P']:
    a=rows(ROOT/f'raw/steps/{f}_a2_b1.jsonl')
    assert len(a)==len(s)==64
    ratios=np.array([x['A_norm']/y['A_norm'] for x,y in zip(a,s)])
    o['norm_trajectory'][f]={'ratio_first':float(ratios[0]),'ratio_last':float(ratios[-1]),
      'ratio_min':float(ratios.min()),'ratio_max':float(ratios.max()),'ratio_mean':float(ratios.mean()),
      'within_branch_norm_match_max_abs':max(abs(x['D_norm']-x['A_norm']) for x in a)}
# all unique logs are internally consecutive and compare global step/cursor/eta across branches
allsteps={p.stem:rows(p) for p in (ROOT/'raw/steps').glob('*.jsonl')}
o['step_log_checks']={'branches':len(allsteps),'unique_rows':sum(map(len,allsteps.values())),
 'all_have_64_consecutive_steps':all([r['branch_step'] for r in v]==list(range(1,65)) for v in allsteps.values()),
 'all_main_cursor_eta_global_step_same':all([(r['cursor'],r['eta'],r['global_step']) for r in v]==[(r['cursor'],r['eta'],r['global_step']) for r in s] for k,v in allsteps.items() if not k.startswith('aux_'))}
o['raw_eval']={'files':len(list((ROOT/'raw/eval').glob('*.jsonl'))),
 'rows':sum(len(rows(p)) for p in (ROOT/'raw/eval').glob('*.jsonl'))}
# digest self-consistency; no archive manifest of all files was provided
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ma=json.loads((ROOT/'run_manifest.json').read_text())
source_files=('train.py','test_v15.py','config.json','RUN_VERIFIED.ps1')
source_hash=hashlib.sha256(json.dumps({n:sha(ROOT/n) for n in source_files},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
o['hash_checks']={'manifest_sha':sha(ROOT/'run_manifest.json'),
 'config':sha(ROOT/'config.json')==ma['config_sha256'],
 'source':source_hash==ma['source_hash'],
 'confirm_views':sha(ROOT/'confirm_views.jsonl')==ma['confirm_views_sha256'],
 'confirm_documents':sha(ROOT/'confirm_documents.csv')==ma['confirm_documents_sha256']}
sel=json.loads((ROOT/'selection_frozen.json').read_text())
o['hash_checks']['dev_selection_files']=all(sha(ROOT/k.replace('\\','/'))==v for k,v in sel['dev_eval_files'].items())
# recompute grid endpoints and one-step vs 64-step b preference on development only
scores=json.loads((ROOT/'dev_scores.json').read_text())
o['dev_grid_recomputed_max_error']=max(abs(nll(ev(k,64,'dev'))-v) for k,v in scores['h64_dev_nll'].items())
o['b_preference_examples']={}
for f in ['S','F','P']:
 o['b_preference_examples'][f]={str(h):nll(ev(f'{f}_a2_b0.5',h,'dev'))-nll(ev(f'{f}_a2_b1',h,'dev')) for h in [1,16,64]}
# resource counts and ledger consistency limitations
rs=rows(ROOT/'logs/resources.jsonl')
o['resources']={'samples':len(rs),'min_available_ram_gib':min(r['available_ram_bytes'] for r in rs)/2**30,
 'max_rss_gib':max(r['rss_bytes'] for r in rs)/2**30,
 'max_cuda_reserved_gib':max(r['cuda_reserved_bytes'] or 0 for r in rs)/2**30,
 'statuses':{x:sum(r['status']==x for r in rs) for x in sorted(set(r['status'] for r in rs))}}
o['ledger']=json.loads((ROOT/'budget_ledger.json').read_text())
# H0 diagnostic wrong state, raw H0 values are retained as invalid labels
h0={p.name:nll(rows(p)) for p in (ROOT/'raw/eval').glob('H0*')}
o['h0_invalid_as_parent_checkpoint']=h0
(OUT/'recomputed_v15.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(o,ensure_ascii=False,indent=2))
