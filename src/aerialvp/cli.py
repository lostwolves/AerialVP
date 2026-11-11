from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config, PROJECT_ROOT
from .gpt import GPTClient


def checkpoint_status(value):
    path = Path(value)
    if path.is_file():
        return {"path": str(path), "exists": True, "weights_complete": path.stat().st_size > 0}
    missing = []
    indices = list(path.rglob('*.safetensors.index.json')) if path.exists() else []
    for index in indices:
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
        missing.extend(str(index.parent / shard) for shard in sorted(shards) if not (index.parent / shard).is_file())
    weights = list(path.rglob('*.safetensors')) + list(path.rglob('pytorch_model*.bin')) if path.exists() else []
    return {"path": str(path), "exists": path.exists(), "weights_complete": bool(weights) and not missing, "missing_shards": missing}


def main():
    parser = argparse.ArgumentParser(description="AerialVP: task engine + MCP tool repository + generated executable program")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "enhance"):
        p = sub.add_parser(name, help="Generate executable program" + (" and run it" if name == "enhance" else " without loading vision models"))
        p.add_argument("--image", required=True)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--prompt")
        group.add_argument("--prompt-file")
        p.add_argument("--run-dir", default=None)
        p.add_argument("--regions", help="Optional JSON object with objects and image_size; original-image pixel xyxy boxes")
    sub.add_parser("list-tools", help="Discover actual MCP servers and JSON schemas")
    sub.add_parser("doctor", help="Inspect configuration and dependencies without API calls")
    sub.add_parser("probe-api", help="Small real calls to the configured engine and analysis models")
    p = sub.add_parser("infer", help="Run the unchanged backend VLM with an already enhanced prompt and original image")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--model")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "doctor":
        packages = {}
        for name in ("mcp", "torch", "torchvision", "transformers", "ultralytics", "accelerate", "bitsandbytes"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        print(json.dumps({"config": config["config_path"], "python": sys.executable,
                          "api_key_configured": bool(os.getenv(config["api"]["api_key_env"])),
                          "engine": config["engine"]["model"], "analysis": config["analysis"]["model"],
                          "packages": packages,
                          "checkpoints": {key: checkpoint_status(val["checkpoint"]) for key, val in config["models"].items()}}, indent=2, ensure_ascii=False))
    elif args.command == "list-tools":
        from .repository import ToolRepository
        async def show():
            async with ToolRepository(config["config_path"]) as repo:
                print(json.dumps(await repo.list_tools(), indent=2, ensure_ascii=False))
        asyncio.run(show())
    elif args.command == "probe-api":
        api = {**config["api"], "timeout": 45, "max_retries": 1, "max_tokens": 32}
        results = []
        for role in ("engine", "analysis"):
            model = config[role]["model"]
            try:
                answer = GPTClient(api, model=model).chat("Reply with OK only.")
                results.append({"role": role, "model": model, "success": True, "answer": answer})
            except RuntimeError as exc:
                results.append({"role": role, "model": model, "success": False, "error": str(exc)})
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if not all(result["success"] for result in results):
            raise SystemExit(1)
    elif args.command == "infer":
        image = Path(args.image).resolve()
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        model = args.model or config["backend"]["model"]
        answer = GPTClient(config["api"], model=model).chat(prompt, image_files=[str(image)])
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"model": model, "image_path": str(image), "prompt": prompt, "answer": answer}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(answer)
    else:
        from .engine import TaskEngine
        prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
        run_dir = args.run_dir or str(PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
        positions = json.loads(Path(args.regions).read_text(encoding="utf-8")) if args.regions else None
        program = asyncio.run(TaskEngine(config["config_path"]).prepare(prompt, args.image, run_dir, positions))
        print(f"Generated executable program: {program}", flush=True)
        if args.command == "enhance":
            subprocess.run([sys.executable, str(program)], check=True)


if __name__ == "__main__":
    main()

