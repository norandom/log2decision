"""Run the frozen fold-zero score and NPU policy on label-blind development inputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time


ARTIFACTS = Path('D:/BuildArtifacts/log2decision')
INTEGRATED_CLI_SHA256 = '9215ee7267a50c257299a1a94a5b2ef2cb0f46f034307c6b7547633272a32c56'
FROZEN_MANIFEST_SHA256 = 'e098d4df5ef4ea623d37789c3347fa4b22845b431ec1e0b0100a7dcb3c43a71e'
GIB = 1024 ** 3
BUDGET = 12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def publish(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def panel(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['text', 'endpoint'], f'Unexpected panel columns in {path}')
        rows = list(reader)
    by_text = {row['text']: row['endpoint'] for row in rows}
    require(len(by_text) == len(rows) and all(by_text), f'Duplicate or empty original in {path}')
    return by_text


def record_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def storage(target):
    size = sum(item.stat().st_size for item in target.rglob('*') if item.is_file())
    return {'target': str(target), 'target_bytes': size, 'free_bytes': shutil.disk_usage(target).free}


def run_policy(run, runtime, output=None):
    repo = Path(__file__).resolve().parents[2]
    run = run.resolve(strict=True)
    artifacts = ARTIFACTS.resolve(strict=True)
    require(artifacts in run.parents, 'Use a directory below the project artifacts root')
    target = (ARTIFACTS / 'task-1.26-target').resolve(strict=True)
    binary = target / 'debug/l2d.exe'
    # A later foreground build may have replaced the reusable debug executable.
    archived_binary = run / f'l2d-{INTEGRATED_CLI_SHA256[:8]}.exe'
    if archived_binary.is_file():
        binary = archived_binary
    require(sha(binary) == INTEGRATED_CLI_SHA256, 'Executable differs from the certified merged CLI')
    query = run / 'queries.csv'
    history = run / 'history.csv'
    marks_file = run / 'training-marks.json'
    manifest = run / 'manifest.json'
    require(sha(manifest) == FROZEN_MANIFEST_SHA256, 'Frozen development plan differs')
    commitments = read_json(manifest)
    inputs = {p.name: sha(p) for p in (history, query, marks_file)}
    for name, digest in inputs.items():
        require(commitments['input_hashes'][name] == digest, f'Frozen input differs: {name}')
    history_rows = panel(history)
    query_rows = panel(query)
    for rows, field in ((history_rows, 'history_ids'), (query_rows, 'query_ids')):
        require({record_id(text) for text in rows} == set(commitments[field]), f'Frozen {field} differ')
    require(history_rows.keys().isdisjoint(query_rows), 'Original overlap between history and queries')
    require(set(history_rows.values()).isdisjoint(query_rows.values()), 'Endpoint overlap')
    marks = read_json(marks_file)
    ordered = list(history_rows)
    marked = {key: set(marks[key]) for key in ('bad', 'good')}
    for key, members in marked.items():
        require(members and len(members) == len(marks[key]) and members <= history_rows.keys(),
                f'Invalid or missing {key} training members')
    require(marked['bad'].isdisjoint(marked['good']), 'Conflicting training marks')

    destination = run
    if output is not None:
        destination = output.resolve()
        require(artifacts in destination.parents, 'Output must be below the project artifacts root')
        destination.mkdir(parents=True, exist_ok=False)
    evidence = Path(tempfile.mkdtemp(prefix='policy-cli-', dir=destination))
    project = evidence / 'project'
    project.mkdir()
    profile_dir = ARTIFACTS / 'task-2.17/npu/f0-s0-mask'
    env = dict(os.environ, L2D_RUNTIME_DIR=str(runtime.resolve(strict=True)),
               L2D_NPU_WORK_DIR=str(ARTIFACTS / 'task-2.19/npu'), CARGO_TARGET_DIR=str(target))

    def command(name, *args, heavy=False):
        argv = list(map(str, [binary, '--project', project, '--memory-limit', '2GB', '--json', *args]))
        admission = storage(target) if heavy else None
        if admission:
            publish(evidence / f'{name}-storage.json', admission)
            require(admission['target_bytes'] < 30 * GIB and admission['free_bytes'] >= 50 * GIB,
                    'Project storage admission failed')
        publish(evidence / f'{name}-command.json', argv)
        print(f'Running {name}', flush=True)
        started = time.monotonic()
        out = evidence / f'{name}.stdout.json'
        with out.open('xb') as stdout, (evidence / f'{name}.stderr').open('xb') as stderr:
            result = subprocess.run(argv, cwd=repo, env=env, stdout=stdout, stderr=stderr, check=False)
        publish(evidence / f'{name}-status.json', {'exit_code': result.returncode,
                                                   'seconds': time.monotonic() - started})
        require(result.returncode == 0, f'{name} failed. See {evidence}')
        return read_json(out)

    for label, members in marked.items():
        # The CSV adapter numbers logical records from 1, including quoted multiline text.
        arguments = [arg for pos, text in enumerate(ordered, 1) if text in members
                     for arg in ('--position', pos)]
        result = command(f'mark-{label}', 'mark', '--input', history, '--no-labels',
                         '--group-column', 'endpoint', '--name', label, '--label', label, *arguments)
        require(result['count'] == len(members) and {r['text'] for r in result['matches']} == members,
                f'Product {label} membership mismatch')
        require(result['source']['content_sha256'] == inputs[history.name], 'Mark source mismatch')
    history_options = ['--history-no-labels', '--history-group-column', 'endpoint']
    scorer = command('fit', 'score', 'fit', '--mark-set', 'bad', '--other-mark-set', 'good',
                     '--name', 'scorer', *history_options, heavy=True)
    require(scorer['confirmed_rules'] == [] and scorer['counts']['marked'] == len(marked['bad'])
            and scorer['counts']['other'] == len(marked['good']), 'Scorer membership mismatch')
    for filename, digest in scorer['model_files_sha256'].items():
        require(sha(Path(scorer['scorer']['directory']) / filename) == digest, 'Saved scorer hash mismatch')
    info = command('profile', 'encoder', 'info', '--profile-dir', profile_dir)
    policy = {'version': 'csic-fold-zero-development-v1',
              'selection_provenance': 'Frozen development comparison. Thresholds and audit seed taken from the task 2.9 engineering fixture. Reservations and review capacity are diagnostic.',
              'low': 0.45, 'high': 0.55, 'corroboration': 1,
              'reach': {'radius': 0.05, 'top_k': 3, 'minimum_labelled_support': 1,
                        'expected_profile': info['profile']},
              'require_calibration': False, 'review_budget': BUDGET, 'unknown_slots': 2,
              'audit_slots': 2, 'audit_seed': 5}
    config = evidence / 'policy.json'
    publish(config, policy)
    options = ['--input', query, '--no-labels', '--group-column', 'endpoint', '--baseline', 'scorer']
    raw = command('scores', 'baseline', 'score', *options)
    raw_scores = {row['text']: row['ratio'] for row in raw['scores']}
    require(raw['scorer']['revision'] == scorer['scorer']['revision'] and raw_scores.keys() == query_rows.keys()
            and len(raw['scores']) == len(raw_scores), 'Score-only panel or scorer differs')

    args = ['triage', *options, *history_options, '--method', 'npu', '--profile-dir', profile_dir,
            '--mark-set', 'bad', '--policy', config]
    report_file = evidence / 'policy-report.json'
    first = command('policy', *args, '--out', report_file, heavy=True)
    second = command('replay', *args, '--replay', report_file, heavy=True)
    require(second['replay_verified'], 'Product policy replay did not verify')
    report = read_json(report_file)
    for output in (first, second):
        record = output['run_record']
        graphs = record['settings']['graph_placement']
        require(record['device'] == 'npu' and record['encoder'] == info['encoder']
                and record['placement']['source'] == 'provider_report'
                and {g['graph'] for g in graphs} == {'line', 'byte'}
                and all(g['npu_nodes'] > 0 and not g['executed_elsewhere'] for g in graphs),
                'Both pinned graphs must execute on the NPU')
        require(record['settings']['mark_source']['content_sha256'] == inputs[history.name]
                and record['inputs'][0]['content_sha256'] == inputs[query.name]
                and record['settings']['history_revision'] == report['evidence_revision'], 'NPU inputs differ')
    require(report['scorer_revision'] == scorer['scorer']['revision'] and report['policy'] == policy
            and report['actual_profile'] == info['profile'] and report['calibration'] == 'Missing',
            'Unexpected policy configuration or scorer')
    require({row['input']['text'] for row in report['decisions']} == query_rows.keys()
            and len(report['decisions']) == len(query_rows), 'Policy and score-only panels differ')
    decisions = []
    for row in report['decisions']:
        text = row['input']['text']
        score = row['score']['diagnostic_ratio']
        require(row['input']['group'] == query_rows[text] and 0 <= score <= 1
                and struct.pack('!f', score) == struct.pack('!f', raw_scores[text]), 'Product score or group mismatch')
        require(not row['score']['comparison_class_assumed'], 'Unexpected assumed comparison class')
        decisions.append({'id': record_id(text), 'score': score,
                          'selected': row['allocation'] is not None, 'queue': row['queue'],
                          'reason': row['reason'], 'allocation': row['allocation'],
                          'supported': row['score']['actionable_ratio'] is not None})
    decisions.sort(key=lambda row: row['id'])
    score_order = sorted(decisions, key=lambda row: (-row['score'], row['id']))
    publish(evidence / 'score-alone-selected.json', [row['id'] for row in score_order[:BUDGET]])
    for name, digest in inputs.items():
        require(sha(run / name) == digest, f'Input changed during run: {name}')
    finish = storage(target)
    metadata = {'evidence_directory': evidence.name, 'binary_sha256': INTEGRATED_CLI_SHA256,
                'manifest_sha256': sha(manifest), 'inputs_sha256': inputs, 'model_sha256': scorer['model_files_sha256'],
                'scorer_revision': report['scorer_revision'], 'history_revision': report['evidence_revision'],
                'profile': info['profile'], 'policy_config_sha256': sha(config), 'report_sha256': sha(report_file),
                'score_alone_selections_sha256': sha(evidence / 'score-alone-selected.json'),
                'allocations': {key: report[key] for key in ('general', 'unknown', 'audit', 'deferred')},
                'replay_verified': True, 'storage_after': finish}
    publish(evidence / 'metadata.json', metadata)
    published = destination / 'policy-decisions.json'
    if published.exists():
        require(read_json(published) == decisions, 'Rerun decisions differ from the frozen publication')
    else:
        publish(published, decisions)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Original frozen development input directory')
    parser.add_argument('--output', type=Path, help='New directory for replay receipts and decisions (default: input directory)')
    parser.add_argument('--runtime', type=Path, default=Path('D:/Source/log2decision/runtime/ryzenai'))
    args = parser.parse_args()
    run_policy(args.run, args.runtime, args.output)
