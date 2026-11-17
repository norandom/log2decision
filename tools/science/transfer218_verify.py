"""Independent identity, replay and non-vacuous NPU export checks for 2.18."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from transfer218 import paired_support


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def verify(root, historical):
    old=json.loads((historical/'data/csic.labels.json').read_text())
    rows=[]
    for fold in range(5):
        directory=root/f'f{fold}/npu'
        provenance=json.loads((directory/'provenance.json').read_text())
        for name,expected in provenance['files'].items():
            assert sha(directory/name)==expected, name
        mapping=json.loads((directory/'mapping.json').read_text())
        with (root/f'f{fold}/paired.jsonl').open(encoding='utf-8') as handle:
            pairs=[json.loads(line) for line in handle]
        expected=[[row['i'],arm] for row in pairs if paired_support(row['raw'],row['redacted']) for arm in ['raw','redacted']]
        assert mapping['rows']==expected
        vectors=np.memmap(directory/'npu-raw.f32',mode='r',dtype='<f4').reshape(-1,128)
        assert len(vectors)==len(mapping['rows'])
        text=[pairs[i][arm] for i,arm in mapping['rows'][:64]]
        unique_vectors=len({v.tobytes() for v in vectors[:64]})
        assert len(set(text))>1 and unique_vectors>1, 'vacuous row equivalence'
        prior=np.memmap(historical/f'enc/f{fold}-s0/mask/csic-int8.f32',mode='r',dtype='<f4').reshape(-1,128)
        ids=[row[0] for row in mapping['rows'][::2]]
        exact=(vectors[::2]==prior[ids]).all(axis=1)
        left=np.array(vectors[::2],dtype=np.float64); right=np.array(prior[ids],dtype=np.float64)
        lengths=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
        valid=lengths>0
        cosine=np.sum(left[valid]*right[valid],axis=1)/lengths[valid]
        if fold==0:
            prior_npu=np.memmap(historical/'npu/f0-s0-mask/npu-raw.f32',mode='r',dtype='<f4').reshape(-1,128)
            assert np.array_equal(vectors[::2],prior_npu[ids])
        train=np.flatnonzero(np.array(old['fold'])!=fold)
        byte_record=json.loads((historical/f'enc/f{fold}-s0/bytes.json').read_text())
        assert len(train)==byte_record['training_lines']
        reconstructed=dict(note='reconstructed from original frozen code/settings, not a new training run',
              source_sha256=old['sha256'],held_fold=fold,training_indices=train.tolist(),
              byte_quantization_indices=np.sort(np.random.default_rng(0).choice(train,256,replace=False)).tolist(),
              line_quantization_indices=np.sort(np.random.default_rng(0).choice(train,512,replace=False)).tolist(),
              threshold_indices=np.sort(np.random.default_rng(1000).choice(train,4096,replace=False)).tolist())
        assert all(old['fold'][i]!=fold for k in ['training_indices','byte_quantization_indices','line_quantization_indices','threshold_indices'] for i in reconstructed[k])
        (root/f'f{fold}/encoder-training-manifest.json').write_text(json.dumps(reconstructed))
        rows.append(dict(fold=fold,vectors=len(vectors),originals=len(ids),first64_distinct_texts=len(set(text)),
                         first64_distinct_vectors=unique_vectors,raw_bit_identical_prior_cpu_reference=int(exact.sum()),
                         cpu_reference_cosine_min=float(cosine.min()),cpu_reference_cosine_p01=float(np.quantile(cosine,.01)),
                         cpu_reference_undefined_cosine=int((~valid).sum()),
                         prior_f0_npu_bit_identity=fold==0,sha256=sha(directory/'npu-raw.f32')))
    (root/'encoding-audit.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('historical',type=Path);a=p.parse_args()
    import memguard
    memguard.limit(4)
    with memguard.hungry('2.18 encoding identity audit'):
        verify(a.root,a.historical)
        print(json.dumps(memguard.peaks()))
