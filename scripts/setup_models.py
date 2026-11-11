"""Download model checkpoints from their official repositories."""
import argparse
import hashlib
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
HF_SOURCES = {
    'spaceom': 'remyxai/SpaceOm',
    'llava_onevision': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    'llava15': 'llava-hf/llava-1.5-7b-hf',
    'grounding_dino': 'IDEA-Research/grounding-dino-tiny',
    'dam': 'nvidia/DAM-3B-Self-Contained',
    'sam': 'facebook/sam-vit-huge',
}
FILE_SOURCES = {
    'yoloworld': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt',
        'filename': 'yolov8x-world.pt',
    },
    'clip': {
        'url': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
        'filename': 'ViT-B-32.pt',
        'sha256': '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source['filename']
    if target.is_file() and target.stat().st_size:
        if not source.get('sha256') or sha256(target) == source['sha256']:
            return target
    partial = target.with_suffix(target.suffix + '.part')
    with requests.get(source['url'], stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with partial.open('wb') as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    if not partial.stat().st_size:
        raise RuntimeError(f'Empty checkpoint download: {target.name}')
    if source.get('sha256') and sha256(partial) != source['sha256']:
        raise RuntimeError(f'Checkpoint checksum mismatch: {target.name}')
    partial.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(FILE_SOURCES) + list(HF_SOURCES), required=True)
    parser.add_argument('--destination', type=Path, default=ROOT / 'models')
    parser.add_argument('--revision', default='main', help='Hugging Face branch, tag, or commit; resolved commits are recorded')
    args = parser.parse_args()
    api = HfApi()
    for name in args.models:
        destination = args.destination / name
        print(f'Downloading {name} to {destination}', flush=True)
        if name in FILE_SOURCES:
            source = FILE_SOURCES[name]
            target = download_file(source, destination)
            manifest = {'source': source['url'], 'filename': target.name, 'sha256': sha256(target)}
        else:
            repo = HF_SOURCES[name]
            info = api.model_info(repo, revision=args.revision)
            snapshot_download(repo_id=repo, revision=info.sha, local_dir=destination,
                allow_patterns=['*.json', '*.safetensors', '*.bin', '*.model', '*.tiktoken', '*.py', '*.txt', '*.jinja', 'README.md', 'LICENSE*'], max_workers=4)
            manifest = {'repository': repo, 'revision': info.sha}
        (destination / 'aerialvp_source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Ready: {name}', flush=True)


if __name__ == '__main__':
    main()
