"""Validation of tool instructions, dependencies, and executable programs (Eqs. 5-9)."""
from __future__ import annotations

import ast
import json
import keyword
import re

import jsonschema

DIMENSIONS = ("spatial_position_description", "semantic_description", "spatial_relationship_description")
TYPE_NAMES = ("Visual Grounding", "Visual Reasoning", "Visual Question Answering")
PLAN_SCHEMA = {
    "type": "object", "required": ["steps", "outputs"], "additionalProperties": False,
    "properties": {
        "steps": {"type": "array", "items": {
            "type": "object", "required": ["id", "tool", "arguments"], "additionalProperties": False,
            "properties": {"id": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"}}
        }},
        "outputs": {"type": "object", "additionalProperties": {"type": "string"}}
    }
}


def resolve(value, context: dict, results: dict):
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                raise ValueError("A reference must be exactly {'$ref': 'ctx.key'} or {'$ref': 'step_id'}")
            name = value["$ref"]
            if name.startswith("ctx."):
                key = name[4:]
                if key not in context:
                    raise ValueError(f"Unknown context reference: {name}")
                return context[key]
            if name not in results:
                raise ValueError(f"Unknown or forward step reference: {name}")
            return results[name]
        return {k: resolve(v, context, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, context, results) for v in value]
    return value


def validate_instruction(instruction: dict, tools: list[dict], allowed: set[str]) -> None:
    if set(instruction) != {"tool", "arguments"}:
        raise ValueError("Instruction must contain only tool and arguments")
    name = instruction["tool"]
    if name not in allowed:
        raise ValueError(f"Tool {name!r} is not available at this stage. Allowed: {sorted(allowed)}")
    spec = next((tool for tool in tools if tool["name"] == name), None)
    if spec is None:
        raise ValueError(f"Tool {name!r} is not registered by MCP")
    jsonschema.validate(instruction["arguments"], spec["inputSchema"])


def validate_plan(plan: dict, context: dict, tools: list[dict], selected: dict) -> None:
    jsonschema.validate(plan, PLAN_SCHEMA)
    requested = set(context["enhancement_plan"])
    if set(plan["outputs"]) != requested:
        raise ValueError(f"outputs must contain exactly the requested dimensions: {sorted(requested)}")
    specs = {tool["name"]: tool for tool in tools}
    allowed_tools = set(selected.values())
    seen = {}
    step_types = {}
    for step in plan["steps"]:
        name = step["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name) or name in {"ctx", "tools", "fuse_enhanced_prompt"}:
            raise ValueError(f"Invalid step id: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate step id: {name}")
        tool = step["tool"]
        if tool not in allowed_tools or tool not in specs:
            raise ValueError(f"Unregistered or unconfigured enhancement tool: {tool}")
        args = resolve(step["arguments"], context, seen)
        jsonschema.validate(args, specs[tool]["inputSchema"])
        # Arguments must stay bound to this task, rather than model-invented coordinates or images.
        if step["arguments"].get("image_path") != {"$ref": "ctx.image_path"}:
            raise ValueError("image_path must reference ctx.image_path")
        if tool == selected["spatial_position_description"]:
            if step["arguments"].get("task_objects") != {"$ref": "ctx.task_objects"}:
                raise ValueError("The detector must use ctx.task_objects")
            seen[name] = {"objects": [], "image_size": [1, 1]}
            step_types[name] = "spatial_position_description"
        else:
            positions = step["arguments"].get("positions")
            position_ref = positions.get("$ref") if isinstance(positions, dict) else None
            if position_ref != "ctx.existing_positions" and step_types.get(position_ref) != "spatial_position_description":
                raise ValueError("Description tools require positions from a preceding detector or ctx.existing_positions")
            if position_ref == "ctx.existing_positions" and not context["existing_positions"].get("objects"):
                raise ValueError("No supplied positions exist; schedule the detector as a prerequisite")
            dim = next(k for k, v in selected.items() if v == tool)
            guidance_key = "semantic_guidance" if dim == "semantic_description" else "relation_guidance"
            if step["arguments"].get("guide_prompt") != {"$ref": f"ctx.{guidance_key}"}:
                raise ValueError(f"Use the paper-derived guide prompt: ctx.{guidance_key}")
            seen[name] = {"descriptions": []}
            step_types[name] = dim
    for dimension, result_id in plan["outputs"].items():
        if result_id == "ctx.existing_positions" and dimension == "spatial_position_description":
            if not context["existing_positions"].get("objects"):
                raise ValueError("Cannot output empty supplied positions")
        elif step_types.get(result_id) != dimension:
            raise ValueError(f"Output {dimension} does not refer to a matching completed tool")
    # No unused tool execution: extra positions are allowed only as a dependency.
    used = set(plan["outputs"].values())
    for step in reversed(plan["steps"]):
        if step["id"] not in used:
            raise ValueError(f"Unused tool step: {step['id']}")
        for ref in _refs(step["arguments"]):
            if not ref.startswith("ctx."):
                used.add(ref)


def _refs(value):
    if isinstance(value, dict):
        if "$ref" in value:
            yield value["$ref"]
        else:
            for item in value.values():
                yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _expression(value) -> str:
    if isinstance(value, dict):
        if "$ref" in value:
            ref = value["$ref"]
            return f"ctx[{ref[4:]!r}]" if ref.startswith("ctx.") else ref
        return "{" + ", ".join(f"{key!r}: {_expression(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_expression(item) for item in value) + "]"
    return repr(value)


def expected_program(plan: dict) -> str:
    lines = ["async def enhance(ctx, tools):"]
    for step in plan["steps"]:
        lines.append(f"    {step['id']} = await tools.call_tool({step['tool']!r}, {_expression(step['arguments'])})")
    outputs = {dim: {"$ref": ref} for dim, ref in plan["outputs"].items()}
    lines.append(f"    return fuse_enhanced_prompt(ctx, {_expression(outputs)})")
    return "\n".join(lines) + "\n"


def validate_program(source: str, plan: dict) -> None:
    """Require the generated function to implement the already validated execution graph.

    Quotes, whitespace and comments can differ. Tool identities, arguments, variable
    dependencies and the fusion call must match; arbitrary generated code is never run.
    """
    if not isinstance(source, str):
        raise ValueError("source must be a Python source string")
    try:
        actual = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Generated program syntax error: {exc.msg} at line {exc.lineno}") from None
    expected = ast.parse(expected_program(plan))
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Generated function does not match the validated tool plan. Use precisely the specified ids, ordered arguments, references and fusion call; no imports or other statements.")

