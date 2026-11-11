"""Run one visual tool model on a user-supplied image."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from aerialvp.config import load_config
from aerialvp.prompts import SEMANTIC_GUIDANCE, RELATION_GUIDANCE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--positions', help='Detection result JSON for description models')
    parser.add_argument('--objects', nargs='+', default=['car'], help='Categories for detection models')
    parser.add_argument('--description', choices=['semantic', 'relation'], default='relation')
    parser.add_argument('--limit', type=int, default=1, help='Explicit candidate limit for this smoke test only')
    args = parser.parse_args()
    config = load_config(args.config)
    directory = Path(config['project_root']) / 'work'
    directory.mkdir(exist_ok=True)
    data = {'image_path': args.image}
    if args.model in ('yoloworld', 'grounding_dino'):
        data['task_objects'] = args.objects
    else:
        if not args.positions:
            parser.error('--positions is required for description models')
        positions = json.loads(Path(args.positions).read_text(encoding='utf-8'))
        positions['objects'] = positions['objects'][:args.limit] if args.limit else positions['objects']
        data.update(positions=positions, guide_prompt=SEMANTIC_GUIDANCE if args.model == 'dam' or args.description == 'semantic' else RELATION_GUIDANCE)
    started = time.monotonic()
    environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    process = subprocess.run([config['models'][args.model]['python'], '-m', 'aerialvp.model_worker', '--config', config['config_path'], '--model', args.model],
                             input=json.dumps(data), text=True, encoding='utf-8', errors='replace', env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (directory / f'smoke_{args.model}.log').write_text(process.stderr, encoding='utf-8')
    if process.returncode:
        print(process.stderr[-6000:])
        raise SystemExit(process.returncode)
    result = json.loads(process.stdout)
    (directory / f'smoke_{args.model}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model': args.model, 'seconds': round(time.monotonic()-started, 2), 'result': result}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
