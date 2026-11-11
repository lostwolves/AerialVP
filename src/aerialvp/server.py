"""Actual MCP stdio servers for prompt analysis and local visual tools."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from .config import load_config

class TaskTypeResult(BaseModel):
    task_type: Literal['Visual Grounding', 'Visual Reasoning', 'Visual Question Answering']

class TaskObjectsResult(BaseModel):
    task_object: list[str] = Field(min_length=1)

class TaskPlanResult(BaseModel):
    task_enhancement_plan: list[Literal['spatial_position_description', 'semantic_description', 'spatial_relationship_description']]

class LocatedObject(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: str
    category: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = None

class PositionResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    objects: list[LocatedObject]
    image_size: list[int] = Field(min_length=2, max_length=2)

class ObjectDescription(BaseModel):
    model_config = ConfigDict(extra='allow')
    object_id: str
    text: str

class DescriptionResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    descriptions: list[ObjectDescription]

def checked(result_type, value):
    try:
        return result_type.model_validate(value)
    except ValueError as exc:
        raise ValueError(f'[AERIALVP_VALIDATION] {exc}') from exc

def execute_model_worker(config, config_path, model_name, arguments):
    if model_name not in config.get('models', {}):
        raise RuntimeError(f'Model implementation {model_name!r} is not configured')
    settings = config['models'][model_name]
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    command = [settings.get('python') or sys.executable, '-m', 'aerialvp.model_worker', '--config', config_path, '--model', model_name]
    try:
        completed = subprocess.run(command, input=json.dumps(arguments, ensure_ascii=False),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
            cwd=Path(__file__).resolve().parents[2], env=env,
            timeout=settings.get('timeout_seconds', 1800), check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Local model {model_name} exceeded its configured timeout') from exc
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode:
        detail = completed.stderr.strip()[-3000:] or f'exit code {completed.returncode}'
        raise RuntimeError(f'Local model {model_name} failed: {detail}')
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f'[AERIALVP_VALIDATION] Local model {model_name} did not produce a JSON result') from exc
    if not isinstance(value, dict):
        raise ValueError(f'[AERIALVP_VALIDATION] Local model {model_name} returned a non-object')
    return value

def create_server(config_path: str, group: str) -> FastMCP:
    path = str(Path(config_path).expanduser().resolve())
    config = load_config(path)
    server = FastMCP(f'AerialVP {group} tools', log_level='WARNING')
    if group == 'analysis':
        from .gpt import GPTClient, parse_object
        from .prompts import build_object_prompt, build_plan_prompt, build_type_prompt
        def analyze(question):
            text = GPTClient(config['api'], model=config['analysis']['model']).chat(question)
            try:
                return parse_object(text)
            except ValueError as exc:
                raise ValueError(f'[AERIALVP_VALIDATION] {exc}') from exc
        @server.tool()
        async def task_type_analysis(original_prompt: str) -> TaskTypeResult:
            """Identify the perception task from the original prompt (paper Fig. 6a)."""
            return checked(TaskTypeResult, await asyncio.to_thread(analyze, build_type_prompt(original_prompt)))
        @server.tool()
        async def task_object_analysis(original_prompt: str) -> TaskObjectsResult:
            """Extract target AND anchor object categories (paper Fig. 6b)."""
            return checked(TaskObjectsResult, await asyncio.to_thread(analyze, build_object_prompt(original_prompt)))
        @server.tool()
        async def task_plan_analysis(original_prompt: str, task_type: str, task_objects: list[str]) -> TaskPlanResult:
            """Select enhancement dimensions according to task and existing information (Fig. 6c)."""
            return checked(TaskPlanResult, await asyncio.to_thread(analyze, build_plan_prompt(original_prompt, task_type, task_objects)))
    elif group == 'vision':
        def position_tool(model):
            async def invoke(image_path: str, task_objects: list[str]) -> PositionResult:
                value = await asyncio.to_thread(execute_model_worker, config, path, model,
                    {'image_path': image_path, 'task_objects': task_objects})
                return checked(PositionResult, value)
            return invoke
        def description_tool(model):
            async def invoke(image_path: str, positions: dict, guide_prompt: str) -> DescriptionResult:
                value = await asyncio.to_thread(execute_model_worker, config, path, model,
                    {'image_path': image_path, 'positions': positions, 'guide_prompt': guide_prompt})
                return checked(DescriptionResult, value)
            return invoke
        for name, model, description in [
            ('spatial_position_description_yoloworld', 'yoloworld', 'YOLO-World-X: detect task-relevant objects; return original-image pixel xyxy boxes.'),
            ('spatial_position_description_groundingdino', 'grounding_dino', 'Grounding-DINO alternative for task-relevant object positions in pixel xyxy format.')]:
            server.add_tool(position_tool(model), name=name, description=description)
        for name, model, description in [
            ('semantic_description_DAM', 'dam', 'DAM: describe localized object appearance and context, using SAM masks.'),
            ('spatial_relationship_description_SpaceOm', 'spaceom', 'SpaceOm: describe localized objects and their inter-object spatial relations.'),
            ('semantic_description_llava15', 'llava15', 'LLaVA-1.5-7B alternative for object semantic descriptions.'),
            ('semantic_description_llava_onevision', 'llava_onevision', 'LLaVA-OneVision-7B alternative for object semantic descriptions.'),
            ('spatial_relationship_description_llava15', 'llava15', 'LLaVA-1.5-7B alternative for contextual spatial relationships.'),
            ('spatial_relationship_description_llava_onevision', 'llava_onevision', 'LLaVA-OneVision-7B alternative for contextual spatial relationships.')]:
            server.add_tool(description_tool(model), name=name, description=description)
    else:
        raise ValueError(f'Unknown group: {group}')
    return server

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--group', choices=['analysis', 'vision'], required=True)
    args = parser.parse_args()
    create_server(args.config, args.group).run(transport='stdio')

if __name__ == '__main__':
    main()
