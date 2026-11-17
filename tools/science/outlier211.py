"""Task 2.11 frozen development-only endpoint-presence proxy; no encoder runs."""
import collections, hashlib, json, mmap, random, struct, sys
from pathlib import Path
SEED = 20261117
KS = [1, 5, 11, 21]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()
def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'plan.json').exists(), 'frozen plan already exists'
    npu=source/'f0/npu'
    inputs=[source/'records.jsonl', npu/'mapping.json', npu/'layout.json', npu/'provenance.json', source/'encoding-audit.json', source/'f0/encoder-training-manifest.json']
    # Explicit whole-object preparation exception, before loading. 32x file bytes
    # bounds JSON/Python containers; vectors stay a read-only mapping.
    estimate=32*sum(p.stat().st_size for p in inputs)
    assert estimate < 1024**3, 'preparation exceeds declared 1 GiB memory allowance'
    rows=[json.loads(line) for line in inputs[0].open(encoding='utf-8')]
    mapping=json.loads(inputs[1].read_text()); layout=json.loads(inputs[2].read_text())
    assert layout['dimension']==128 and len(rows)==25608
    assert mapping['rows']==[[r['i'],arm] for r in rows for arm in ['raw','redacted']]
    assert not mapping['unsupported']
    provenance=json.loads(inputs[3].read_text())
    for name, expected in provenance['files'].items(): assert sha(npu/name)==expected, name
    actual_vector_sha=sha(npu/'npu-raw.f32')
    assert actual_vector_sha==json.loads(inputs[4].read_text())[0]['sha256']
    with (npu/'texts.jsonl').open(encoding='utf-8') as texts:
        for r in rows:
            assert json.loads(next(texts))==r['text']
            next(texts)
        assert next(texts,None) is None
    groups=collections.defaultdict(list)
    for r in rows: groups[r['group']].append(r['i'])
    rng=random.Random(SEED)
    eligible=sorted(g for g,v in groups.items() if len(v)>=18)
    rng.shuffle(eligible)
    assert len(eligible)>=8, 'insufficient endpoint groups for frozen panel'
    rare_groups=eligible[:3]; new_groups=eligible[3:6]
    reference=[]; queries=[]
    for kind, panel in [('rare',rare_groups),('new',new_groups)]:
        for group in panel:
            ids=groups[group].copy(); rng.shuffle(ids)
            if kind=='rare': reference.extend(ids[:2]); ids=ids[2:]
            queries.extend({'i':i,'proxy':kind,'group':group} for i in ids[:16])
    rest=[r['i'] for r in rows if r['group'] not in rare_groups+new_groups]
    rng.shuffle(rest); reference+=rest[:2048-len(reference)]
    assert len(reference)==2048 and len(queries)==96
    assert len(set(reference))==2048 and not set(reference)&{q['i'] for q in queries}
    train=set(json.loads(inputs[5].read_text())['training_indices'])
    revision=hashlib.sha256((sha(inputs[2])+sha(inputs[3])+':original-raw-f32-no-extra-normalization').encode()).hexdigest()
    plan=dict(seed=SEED,k=KS,development_only=True,metric='squared_l2_distance',
        proxy_definition={'rare':'endpoint has exactly two distinct original reference texts','new':'endpoint absent from reference'},
        limitations=['not temporal novelty, corpus rarity, attack relevance or new human labels','encoder may have seen development queries; no qualification claim','no operational threshold selection'],
        reference=reference,queries=queries,rare_groups=rare_groups,new_groups=new_groups,
        encoder_training_overlap={kind:sum(q['i'] in train for q in queries if q['proxy']==kind) for kind in ['rare','new']},
        reference_training_overlap=sum(i in train for i in reference),
        coverage={'corpus_originals':len(rows),'reference':len(reference),'query':len(queries),'rare_groups':3,'new_groups':3},
        origin={'name':'csic-f0-s0-mask-original-raw-128','revision':revision,'dimension':128,'runtime_release':'1.9.0','onnx_runtime':'1.30.0','driver':'32.0.20102.4480'},
        input_hashes={str(p):sha(p) for p in inputs+[npu/'npu-raw.f32',npu/'texts.jsonl']},
        vector_layout='128 f32 little endian per row; raw/redacted INTERLEAVED; raw vector row 2*i validated against mapping and original text',
        preparation_estimated_bytes=estimate,
        analysis='ROC AUC(new positive), medians, top32 new fraction; endpoint-cluster bootstrap k minus5 with 2000 draws seed20261117; descriptive only')
    # Freeze settings and membership BEFORE any vector distance is computed.
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    lookup={q['i']:q for q in queries}
    with (npu/'npu-raw.f32').open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped, (out/'panel.jsonl').open('w',encoding='utf-8') as dest:
        for i in reference+[q['i'] for q in queries]:
            vector=struct.unpack_from('<128f',mapped,2*i*128*4)
            import math
            assert all(math.isfinite(x) for x in vector) and any(vector)
            r=rows[i]
            dest.write(json.dumps(dict(i=i,text=r['text'],group=r['group'],proxy=lookup[i]['proxy'] if i in lookup else 'reference',vector=vector))+'\n')
    (out/'frozen.json').write_text(json.dumps({'plan_sha256':sha(out/'plan.json'),'panel_sha256':sha(out/'panel.jsonl')},indent=2))
    print(json.dumps(plan['coverage']|{'training_overlap':plan['encoder_training_overlap'],'plan_sha256':sha(out/'plan.json')}))
def auc(pairs):
    # Ties contribute half; fixed small query panel, no vector distance matrix.
    a=[s for y,s in pairs if y]; b=[s for y,s in pairs if not y]
    return sum((x>z)+.5*(x==z) for x in a for z in b)/(len(a)*len(b))
def report(out):
    frozen=json.loads((out/'frozen.json').read_text()); plan=json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json')==frozen['plan_sha256'] and sha(out/'panel.jsonl')==frozen['panel_sha256']
    import statistics
    results={}; scores={}
    for k in KS:
        result=json.loads((out/f'k{k}.json').read_text()); distances={r['query']:r['distance'] for r in result['rows']}
        assert result['distance_metric']=='squared_l2_distance' and result['k']==k and len(distances)==96
        scores[k]=[(q['proxy']=='new',distances[i]) for i,q in enumerate(plan['queries'])]
        results[k]={'auc_new_vs_rare':auc(scores[k]),'median':{kind:statistics.median(distances[i] for i,q in enumerate(plan['queries']) if q['proxy']==kind) for kind in ['rare','new']},'top32_new_fraction':sum(scores[k][r['query']][0] for r in result['rows'][:32])/32}
    rng=random.Random(SEED); deltas={k:[] for k in KS}
    for _ in range(2000):
        indices=[]
        for kind in ['rare','new']:
            groups=plan[kind+'_groups']
            for group in rng.choices(groups,k=len(groups)):
                indices.extend(i for i,q in enumerate(plan['queries']) if q['group']==group)
        base=auc([scores[5][i] for i in indices])
        for k in KS: deltas[k].append(auc([scores[k][i] for i in indices])-base)
    for k in KS:
        delta=sorted(deltas[k]); results[k]['paired_endpoint_bootstrap_auc_delta_vs5_95pct']=[delta[50],delta[1949]]
    (out/'report.json').write_text(json.dumps({'plan_sha256':frozen['plan_sha256'],'results':results,'scope':'constructed development endpoint-presence proxy only'},indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare(Path(sys.argv[2]),Path(sys.argv[3]))
    elif sys.argv[1]=='report': report(Path(sys.argv[2]))
