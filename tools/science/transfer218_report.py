"""Evaluate frozen 2.18 outputs against independent CSIC labels; never tune a model."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from transfer218 import metrics, digest


def readlines(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def run(root):
    records=readlines(root/'records.jsonl'); n=len(records)
    assert n<=30000
    assert len({r['id'] for r in records})==n
    plan=json.loads((root/'plan.json').read_text())
    query=re.compile(plan['seed_query'])
    marks=np.array([bool(query.search(r['text'])) for r in records])
    labels=np.array([r['bad'] for r in records]); identities=[r['id'] for r in records]
    all_results=[]; pooled={}; predictions=[]
    for fold in range(5):
        folder=root/f'f{fold}'
        splits=json.loads((folder/'split.json').read_text())
        assert [r['id'] for r in splits]==identities
        partition=np.array([r['partition'] for r in splits])
        train=partition=='train'
        held=partition=='held'
        assert not ({r['group'] for r in records if train[r['i']]} & {r['group'] for r in records if held[r['i']]})
        frozen=json.loads((folder/'frozen.json').read_text())
        assert int(marks[train].sum())==frozen['counts']['marked']
        product_marks=json.loads((Path(frozen['raw'])/'mark-set.json').read_text())
        assert product_marks['positions']==(np.flatnonzero(marks[train])+1).tolist()
        baseline=readlines(folder/'baseline.jsonl')
        paired=readlines(folder/'paired.jsonl')
        assert [r['i'] for r in paired]==list(range(n))==[r['i'] for r in baseline]
        mapping=json.loads((folder/'npu/mapping.json').read_text())
        vector=np.memmap(folder/'npu/npu-raw.f32',dtype='<f4',mode='r').reshape(-1,128)
        assert len(vector)==len(mapping['rows'])
        supported=np.zeros(n,dtype=bool)
        support_ids=np.array([row[0] for row in mapping['rows'][::2]])
        assert mapping['rows'][::2]==[[int(i),'raw'] for i in support_ids]
        assert mapping['rows'][1::2]==[[int(i),'redacted'] for i in support_ids]
        supported[support_ids]=True
        assert set(mapping['unsupported'])==set(np.flatnonzero(~supported).tolist())
        # Both arms must share support, including invalid/zero vectors.
        norms=np.linalg.norm(vector.astype(np.float64),axis=1)
        valid=np.isfinite(vector).all(axis=1)&(norms>1e-12)
        paired_valid=valid[::2]&valid[1::2]
        supported[support_ids[~paired_valid]]=False
        support_ids=support_ids[paired_valid]
        transformed_train={arm:{digest(paired[i][arm]) for i in np.flatnonzero(train)} for arm in ['raw','redacted']}
        paired_collision=np.array([any(digest(paired[i][arm]) in transformed_train[arm] for arm in ['raw','redacted']) for i in range(n)])
        result=dict(fold=fold,train=int(train.sum()),fitted_report=int((partition=='fitted_report').sum()),
                    held=int(held.sum()),marks=int(marks[train].sum()),rules=len(frozen['rules']),
                    seed_query_bad=int(labels[marks].sum()),seed_query_good=int((1-labels[marks]).sum()),
                    unsupported_original_or_redacted=len(mapping['unsupported']),invalid_vector_pairs=int((~paired_valid).sum()),
                    arms={})
        for arm_index,arm in enumerate(['raw','redacted']):
            x=np.zeros((n,128),dtype=np.float64)
            x[support_ids]=vector[arm_index::2][paired_valid]/norms[arm_index::2][paired_valid,None]
            training=train&supported
            model=LogisticRegression(C=1,max_iter=1000,solver='lbfgs',random_state=20261117)
            model.fit(x[training],marks[training])
            assert model.n_iter_[0]<1000, 'nonconverged frozen diagnostic head'
            # Persist coefficients and training identity before evaluation scoring.
            np.savez(folder/f'head-{arm}.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_,n_iter=model.n_iter_)
            (folder/f'head-{arm}.json').write_text(json.dumps(dict(C=1,max_iter=1000,query=plan['seed_query'],
                training_ids=[identities[i] for i in np.flatnonzero(training)],labels='regex versus assumed unmarked',
                seed=20261117),indent=2))
            score=model.predict_proba(x)[:,1]
            for method,values,coverage in [('baseline',np.array([r[arm] for r in baseline]),np.ones(n,dtype=bool)),('vector_diagnostic',score,supported)]:
                key=method+'_'+arm
                result['arms'][key]={}
                for part in ['fitted_report','held']:
                    subset=partition==part
                    indices=np.flatnonzero(subset)
                    ids=[identities[i] for i in indices]
                    ys=labels[indices].astype(int).tolist()
                    ss=[float(values[i]) if coverage[i] else None for i in indices]
                    output=metrics(ys,ss,ids,100)
                    beyond=indices[~marks[indices]]
                    output['beyond_seed']=metrics(labels[beyond].astype(int).tolist(),[float(values[i]) if coverage[i] else None for i in beyond],[identities[i] for i in beyond],100)
                    # Audit transformed collisions without silently changing the main cohort.
                    collision=np.array([digest(paired[i][arm]) in transformed_train[arm] for i in indices])
                    output['transformed_training_collisions']=int(collision.sum())
                    output['paired_collision_exclusions']=int(paired_collision[indices].sum())
                    clean=indices[~paired_collision[indices]]
                    output['collision_excluded_sensitivity']=metrics(labels[clean].astype(int).tolist(),[float(values[i]) if coverage[i] else None for i in clean],[identities[i] for i in clean],100)
                    result['arms'][key][part]=output
                    target=pooled.setdefault((key,part),dict(labels=[],scores=[],ids=[]))
                    target['labels']+=ys; target['scores']+=ss; target['ids'] += [f'f{fold}:{i}' for i in ids]
                    predictions += [dict(fold=fold,method=key,partition=part,id=i,label=y,score=s) for i,y,s in zip(ids,ys,ss)]
                result['arms'][key]['gap']=result['arms'][key]['fitted_report']['roc_auc']-result['arms'][key]['held']['roc_auc']
        all_results.append(result)
    rng=np.random.default_rng(20261117)
    draws=rng.integers(0,5,size=(10000,5))
    summary={}
    for method in all_results[0]['arms']:
        gap=np.array([r['arms'][method]['gap'] for r in all_results])
        summary[method]=dict(gap_mean=float(gap.mean()),gap_sd=float(gap.std(ddof=1)),
             gap_bootstrap95=np.quantile(gap[draws].mean(axis=1),[.025,.975]).tolist(),
             folds_above_diagnostic_margin=int((gap>.05).sum()),
             fitted_mean=float(np.mean([r['arms'][method]['fitted_report']['roc_auc'] for r in all_results])),
             held_mean=float(np.mean([r['arms'][method]['held']['roc_auc'] for r in all_results])))
        for part in ['fitted_report','held']:
            for metric in ['roc_auc','pr_auc','precision_at_budget','recall_at_budget']:
                values=np.array([r['arms'][method][part][metric] for r in all_results])
                summary[method][part+'_'+metric]=dict(mean=float(values.mean()),sd=float(values.std(ddof=1)))
    changes={}
    for method in ['baseline','vector_diagnostic']:
        difference=np.array([r['arms'][method+'_redacted']['gap']-r['arms'][method+'_raw']['gap'] for r in all_results])
        changes[method]=dict(redacted_minus_raw_gap_mean=float(difference.mean()),sd=float(difference.std(ddof=1)),
                             paired_fold_bootstrap95=np.quantile(difference[draws].mean(axis=1),[.025,.975]).tolist())
    report=dict(status='exploratory; qualification incomplete',plan=plan,folds=all_results,summary=summary,changes=changes,
                pooled={key+'_'+part:metrics(v['labels'],v['scores'],v['ids'],500) for (key,part),v in pooled.items()},
                uncertainty='5 fold descriptive SD; 10,000 paired fold bootstrap draws seed20261117; overlapping training sets, not independent qualification confidence',
                pooled_budget='500 pooled records; additionally each fold has fixed100, pooled ranking does not enforce fold quotas',
                unmeasured_workflows=['pattern-assisted review policy','production score with vector evidence policy'],
                unresolved_conflicts=0,conflict_handling='source duplicate-label conflicts refuse preparation',
                final_holdout='not allocated/unavailable; no final material examined')
    (root/'report.json').write_text(json.dumps(report,indent=2))
    with (root/'predictions.jsonl').open('w') as handle:
        for row in predictions: handle.write(json.dumps(row)+'\n')
    print(json.dumps(dict(summary=summary,changes=changes),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    import memguard
    memguard.limit(4)
    with memguard.hungry('2.18 exploratory frozen report') as guard, threadpool_limits(limits=4):
        run(a.root)
        print(json.dumps(dict(memory=memguard.peaks(),waited=guard.waited,spill_files=0)))
