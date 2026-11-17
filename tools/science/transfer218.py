"""Frozen exploratory transfer experiment; development only, never final qualification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SEED_QUERY = r"(?i)(union(\s|\+|%20)+select|drop(\s|\+|%20)+table|<script|%3cscript)"


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def preflight_source(source, expected_hash, limit=64*1024**2):
    if source.stat().st_size > limit:
        raise ValueError('source exceeds bounded CSIC preparation admission')
    with source.open('rb') as handle:
        actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual != expected_hash:
        raise ValueError('source differs from frozen historical CSIC identity')
    return actual


def partition(text, endpoint_fold, fold):
    if endpoint_fold == fold:
        return 'held'
    return 'fitted_report' if int(digest(text)[:8], 16) % 5 == 0 else 'train'


def paired_support(original, redacted):
    return 0<len(original.encode())<=1024 and 0<len(redacted.encode())<=1024


def metrics(labels, scores, identities, budget):
    from sklearn.metrics import roc_auc_score, average_precision_score
    supported = [i for i, s in enumerate(scores) if s is not None]
    ranked = sorted(supported, key=lambda i: (-scores[i], identities[i]))[:budget]
    positives = sum(labels)
    hits = sum(labels[i] for i in ranked)
    y = [labels[i] for i in supported]
    s = [scores[i] for i in supported]
    both = len(set(y)) == 2
    return dict(records=len(labels), positives=positives, supported=len(supported),
                coverage=len(supported)/len(labels) if labels else None,
                unsupported_positives=sum(labels[i] for i in range(len(labels)) if scores[i] is None),
                roc_auc=float(roc_auc_score(y, s)) if both else None,
                pr_auc=float(average_precision_score(y, s)) if both else None,
                undefined_reason=None if both else 'supported partition does not contain both classes',
                pr_definition='average precision (step integral)', budget=budget,
                reviewed=len(ranked), precision_at_budget=hits/len(ranked) if ranked else None,
                recall_at_budget=hits/positives if positives else None,
                missed_positives=positives-hits, selected_ids=[identities[i] for i in ranked])


def prepare(source, historical, output):
    import l2dsci
    old = json.loads((historical/'data/csic.labels.json').read_text())
    preflight_source(source,old['sha256'])
    output.mkdir(parents=True, exist_ok=False)
    # Bounded whole-object preparation exception: <=64MiB, verified CSIC identity;
    # Windows process commit limit4GiB applies before importing/parsing this corpus.
    data = l2dsci.read_csic(str(source))
    assert data.sha256 == old['sha256']
    folds = l2dsci.cv5_fold_of_group(data)
    assert [folds[g] for g in data.groups] == old['fold']
    assert len(data.texts) <= 30000 and sum(len(t.encode()) for t in data.texts) < 64*1024**2
    # Before any scores: deterministic hash split; seed query and confirmation policy fixed.
    plan = dict(schema=1, source_sha256=data.sha256, seed_query=SEED_QUERY,
                split='original five endpoint folds; sha256(text) first32bits modulo5==0 fitted report',
                confirmation='only proposed localhost:8080 or exact training URL path; no payload tokens',
                baseline=dict(buckets=65536, ngram_max=3, l2=1, max_iter=1000, tolerance=0.001),
                vector_head='L2 logistic regression C=1 max_iter=1000 on regex labels; no tuning',
                review_budget=100, budget_scope='100 distinct records per fold/report partition/workflow',
                diagnostic_gap=0.05, operational_targets=None, final_holdout='not allocated/unavailable',
                temporal='untested: CSIC has no suitable event-time sequence',
                evaluation_labels='CSIC classification independent of seed; no new human review claimed',
                encoder_fitted_report_overlap='existing frozen unsupervised encoder saw fitted-report texts; disclosed optimistic diagnostic',
                pooled_fitted='same records recur in four fold contexts; not independent observations',
                records=data.records, distinct=len(data.texts))
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    with (output/'records.jsonl').open('w', encoding='utf-8') as handle:
        for i, (text, bad, group) in enumerate(zip(data.texts, data.bad, data.groups)):
            handle.write(json.dumps(dict(i=i, id=digest(text), text=text, bad=int(bad), group=group,
                                         fold=folds[group], bytes=len(text.encode())), ensure_ascii=False)+'\n')
    # Preserve source format and only the first occurrence of each training text.
    for fold in range(5):
        folder=output/f'f{fold}'
        folder.mkdir()
        train={t for t,g in zip(data.texts,data.groups) if partition(t,folds[g],fold)=='train'}
        manifest=[dict(i=i,id=digest(t),partition=partition(t,folds[g],fold))
                  for i,(t,g) in enumerate(zip(data.texts,data.groups))]
        (folder/'split.json').write_text(json.dumps(manifest))
        with source.open(encoding='utf-8',newline='') as src, (folder/'train.csv').open('w',encoding='utf-8',newline='') as dst:
            reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames); writer.writeheader()
            for row in reader:
                url=row['URL'].strip(); head,_,version=url.rpartition(' ')
                if head and version.startswith('HTTP/'): url=head
                text=f"{row['Method'].strip()} {url}"+('' if row['content']=='' else '\n'+row['content'])
                if text in train:
                    writer.writerow(row); train.remove(text)
        assert not train
    print(json.dumps(plan,indent=2))


def fixtures(root, historical, models):
    import shutil
    import numpy as np
    import hg17
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for fold in range(5):
        folder=root/f'f{fold}'; out=folder/'npu'; out.mkdir()
        model=models/f'f{fold}-s0'
        byte_record=json.loads((historical/f'enc/f{fold}-s0/bytes.json').read_text())
        line_record=json.loads((historical/f'enc/f{fold}-s0/mask/line.json').read_text())
        assert byte_record['fold']==fold and line_record['fold']==fold
        assert sha(model/'byte.int8.onnx')==byte_record['files']['byte_int8_sha256']
        assert sha(model/'mask/line.int8.onnx')==line_record['files']['line_int8_sha256']
        assert sha(model/'table.bin')==byte_record['files']['table_sha256']
        for src,name in [(model/'byte.int8.onnx','byte.int8.onnx'),(model/'mask/line.int8.onnx','line.int8.onnx'),(model/'table.bin','table.bin')]:
            shutil.copyfile(src,out/name)
        scale,zero=hg17.dequantize_scale(str(out/'byte.int8.onnx'),'y')
        assert zero==0
        hg17.exp_table(scale).astype('<f8').tofile(out/'exp-table.f64')
        layout=json.loads((historical/'npu/f0-s0-mask/layout.json').read_text())
        layout.update(variant=f'f{fold}-s0-mask-transfer218',patcher_setting=byte_record['patching']['threshold'],logit_scale=scale)
        ids=[]; unsupported=[]
        with (out/'texts.jsonl').open('w',encoding='utf-8') as dst, (folder/'paired.jsonl').open(encoding='utf-8') as src:
            for line in src:
                row=json.loads(line)
                # Same original membership for both arms; redaction cannot rescue long originals.
                assert row['original_bytes']==len(row['raw'].encode())
                support=paired_support(row['raw'],row['redacted'])
                if not support:
                    unsupported.append(row['i']); continue
                for arm in ['raw','redacted']:
                    dst.write(json.dumps(row[arm],ensure_ascii=False)+'\n')
                    ids.append([row['i'],arm])
        layout.update(texts=len(ids),csic_texts=len(ids))
        (out/'layout.json').write_text(json.dumps(layout,indent=2))
        (out/'mapping.json').write_text(json.dumps(dict(rows=ids,unsupported=unsupported)))
        provenance=dict(fold=fold,seed=0,variant='mask',training=byte_record['settings']['trained_on'],
                        calibration='original training-only rows: byte256, line512, patch threshold4096; no recalibration',
                        historical_byte_record=sha(historical/f'enc/f{fold}-s0/bytes.json'),
                        historical_line_record=sha(historical/f'enc/f{fold}-s0/mask/line.json'),
                        files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('five fold-correct paired NPU fixtures frozen')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','fixtures']); p.add_argument('source',type=Path); p.add_argument('historical',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args()
    import memguard
    memguard.limit(4)
    with memguard.hungry('2.18 '+a.command) as guard:
        if a.command=='prepare': prepare(a.source,a.historical,a.output)
        else: fixtures(a.output,a.historical,a.source)
        print(json.dumps(dict(memory=memguard.peaks(),waited=guard.waited,external_npu_usage='unknown')))
