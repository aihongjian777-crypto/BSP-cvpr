import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'revision_20261019'
BACKUP = OUT / 'original'
OUT.mkdir(parents=True, exist_ok=True)
paths = [ROOT / x for x in ['README.md', 'config.py', 'train.py', 'train_ct.py',
                            'test.py', 'test_ct.py', 'boundary_loss.py', 'utils.py']]
paths += list((ROOT / 'models').glob('*.py')) + list((ROOT / 'dataloaders').glob('*.py'))
paths += list((ROOT / 'scripts').glob('*.sh'))
manifest = {}
for path in paths:
    rel = path.relative_to(ROOT).as_posix()
    target = BACKUP / rel
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(target))
    original = target.read_bytes()
    record = {'sha256': hashlib.sha256(original).hexdigest(), 'bytes': len(original)}
    if path.suffix == '.py':
        tree = ast.parse(original.decode('utf-8-sig'))
        record['imports'] = [ast.get_source_segment(original.decode('utf-8-sig'), n)
                             for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        record['duplicate_methods'] = {}
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            methods = {}
            for n in cls.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.setdefault(n.name, []).append(n.lineno)
            record['duplicate_methods'][cls.name] = {k: v for k, v in methods.items() if len(v) > 1}
        record['comment_lines'] = sum(x.lstrip().startswith('#') for x in original.decode('utf-8-sig').splitlines())
    manifest[rel] = record
(OUT / 'original_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

runs = []; protected = {}
for folder in ['exps', 'exps_on_CMR', 'checkpoints']:
    for path in (ROOT / folder).rglob('*'):
        if not path.is_file(): continue
        if path.suffix in ['.json', '.log', '.txt', '.pth', '.py']:
            protected[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size,
                                                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.name != 'config.json': continue
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        run_path = path.parent / 'run.json'; metrics_path = path.parent / 'metrics.json'
        run = json.loads(run_path.read_text(encoding='utf-8-sig')) if run_path.exists() else {}
        metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig')) if metrics_path.exists() else {}
        last_steps = {key: max(value.get('steps', [-1]) or [-1]) for key, value in metrics.items() if isinstance(value, dict)}
        logs = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in path.parent.glob('*.log'))
        runs.append({'directory': path.parent.relative_to(ROOT).as_posix(), 'config': config,
                     'status': run.get('status'), 'result': run.get('result'), 'last_metric_steps': last_steps,
                     'evaluation_lines': [s for s in logs.splitlines() if re.search(r'Total Mean Dice|Mean class Dice|Mean class IoU|Tested query volume', s)],
                     'has_IFR_source': 'def QRR_SWPA' in (path.parent / 'source/models/bsp.py').read_text(encoding='utf-8', errors='replace') if (path.parent / 'source/models/bsp.py').exists() else None})
(OUT / 'experiment_audit.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'protected_manifest.json').write_text(json.dumps(protected, ensure_ascii=False, indent=2), encoding='utf-8')
print('Source backups:', len(manifest), 'protected historical files:', len(protected))
for r in runs:
    c = r['config']
    print(r['directory'], 'fold=', c.get('eval_fold'), 'status=', r['status'],
          'logged_max_step=', max(r['last_metric_steps'].values(), default=-1),
          'evaluation_lines=', len(r['evaluation_lines']), 'IFR=', r['has_IFR_source'])
