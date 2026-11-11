"""Execute a generated enhancement program through MCP; fuse only requested outputs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .prompts import fuse_prompt
from .repository import ToolRepository


def fuse_enhanced_prompt(context: dict, information: dict) -> str:
    if set(information) != set(context["enhancement_plan"]):
        raise ValueError("Fusion outputs must match the task enhancement plan exactly")
    blocks = []
    for dimension in context["enhancement_plan"]:
        value = information[dimension]
        if dimension == "spatial_position_description":
            entries = value["objects"]
            text = "; ".join(f"{item['id']} ({item['category']}): {tuple(item['bbox'])}" for item in entries)
            blocks.append("Spatial position description (pixel xyxy coordinates in the original image): " + (text or "No candidate objects detected."))
        else:
            entries = value["descriptions"]
            text = "\n".join(f"{item['object_id']}: {item['text']}" for item in entries)
            label = "Semantic description" if dimension == "semantic_description" else "Spatial relationship description"
            blocks.append(label + ": " + (text or "No descriptions available."))
    return fuse_prompt(context["original_prompt"], context["task_type"], context["task_objects"], "\n".join(blocks))


class RecordingTools:
    def __init__(self, repository, directory: Path):
        self.repository = repository
        self.directory = directory
        self.records = []

    async def call_tool(self, name, arguments):
        result = await self.repository.call_tool(name, arguments)
        self.records.append({"tool": name, "arguments": arguments, "result": result})
        (self.directory / "tool_outputs.json").write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


async def execute_generated(program_path, enhance, output_dir=None, config_path=None):
    directory = Path(program_path).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    destination = Path(output_dir).resolve() if output_dir else directory
    destination.mkdir(parents=True, exist_ok=True)
    async with ToolRepository(config_path or manifest["config_path"]) as repo:
        recorder = RecordingTools(repo, destination)
        prompt = await enhance(manifest["context"], recorder)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enhancement program did not return a nonempty prompt")
    (destination / "enhanced_prompt.txt").write_text(prompt, encoding="utf-8")
    (destination / "enhancement_result.json").write_text(json.dumps({
        "original_prompt": manifest["context"]["original_prompt"],
        "image_path": manifest["context"]["image_path"],
        "enhanced_prompt": prompt,
        "task_type": manifest["context"]["task_type"],
        "enhancement_plan": manifest["context"]["enhancement_plan"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt


def run_generated(program_path, enhance):
    parser = argparse.ArgumentParser(description="Execute this AerialVP-generated program. No GPT planning calls are repeated.")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", help="Override MCP/model deployment config")
    args = parser.parse_args()
    print(asyncio.run(execute_generated(program_path, enhance, args.output_dir, args.config)))

