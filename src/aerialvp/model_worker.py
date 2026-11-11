"""One isolated model invocation; stdout is exclusively the JSON result."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    wire_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .config import load_config
            config = load_config(args.config)
            cache = Path(config['project_root']) / '.cache'
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('HF_HOME', str(cache / 'huggingface'))
            os.environ.setdefault('YOLO_CONFIG_DIR', str(cache / 'ultralytics'))
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            from .model_adapters import run_model
            arguments = json.load(sys.stdin)
            result = run_model(args.model, config['models'][args.model], arguments)
        json.dump(result, wire_stdout, ensure_ascii=False, allow_nan=False)
        wire_stdout.write('\n')
        wire_stdout.flush()
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        json.dump({'error': {'model': args.model, 'type': type(error).__name__, 'message': str(error)}}, wire_stdout, ensure_ascii=False)
        wire_stdout.write('\n')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
