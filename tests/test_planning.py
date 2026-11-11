import copy
import pytest

from aerialvp.planning import expected_program, validate_plan, validate_program
from aerialvp.runtime import fuse_enhanced_prompt


@pytest.fixture
def task():
    context = {"original_prompt": "Find the car nearest the brown car.", "image_path": "image.jpg", "task_type": "Visual Grounding",
               "task_objects": ["car"], "enhancement_plan": ["semantic_description"],
               "existing_positions": {"objects": []}, "semantic_guidance": "Describe attributes.", "relation_guidance": "Describe relations."}
    selected = {"spatial_position_description": "detector", "semantic_description": "semantic", "spatial_relationship_description": "relation"}
    def schema(properties):
        return {"type": "object", "required": list(properties), "properties": properties, "additionalProperties": False}
    tools = [{"name": "detector", "inputSchema": schema({"image_path": {"type": "string"}, "task_objects": {"type": "array"}})},
             {"name": "semantic", "inputSchema": schema({"image_path": {"type": "string"}, "positions": {"type": "object"}, "guide_prompt": {"type": "string"}})}]
    plan = {"steps": [{"id": "positions", "tool": "detector", "arguments": {"image_path": {"$ref": "ctx.image_path"}, "task_objects": {"$ref": "ctx.task_objects"}}},
                      {"id": "semantics", "tool": "semantic", "arguments": {"image_path": {"$ref": "ctx.image_path"}, "positions": {"$ref": "positions"}, "guide_prompt": {"$ref": "ctx.semantic_guidance"}}}],
            "outputs": {"semantic_description": "semantics"}}
    return context, selected, tools, plan


def test_dependency_detector_is_allowed_without_position_fusion(task):
    ctx, selected, tools, plan = task
    validate_plan(plan, ctx, tools, selected)
    validate_program(expected_program(plan), plan)


@pytest.mark.parametrize("change", ["forward_ref", "fabricated_bbox", "wrong_image", "extra_output", "unregistered", "unused"])
def test_bad_tool_graph_is_rejected(task, change):
    ctx, selected, tools, plan = task
    plan = copy.deepcopy(plan)
    if change == "forward_ref":
        plan["steps"].reverse()
    elif change == "fabricated_bbox":
        plan["steps"][1]["arguments"]["positions"] = {"objects": [{"bbox": [1, 2, 3, 4]}]}
    elif change == "wrong_image":
        plan["steps"][0]["arguments"]["image_path"] = "other.jpg"
    elif change == "extra_output":
        plan["outputs"]["spatial_position_description"] = "positions"
    elif change == "unregistered":
        plan["steps"][0]["tool"] = "imaginary_detector"
    elif change == "unused":
        plan["steps"].append({**plan["steps"][0], "id": "unused"})
    with pytest.raises(ValueError):
        validate_plan(plan, ctx, tools, selected)


def test_program_cannot_change_the_validated_plan(task):
    *_, plan = task
    source = expected_program(plan)
    with pytest.raises(ValueError):
        validate_program(source.replace("'detector'", "'unregistered_tool'"), plan)
    with pytest.raises(ValueError):
        validate_program("import os\n" + source, plan)


@pytest.mark.asyncio
async def test_generated_function_executes_dependencies_and_preserves_instruction(task):
    ctx, _, _, plan = task
    calls = []
    positions = {"objects": [{"id": "car_1", "category": "car", "bbox": [1, 2, 3, 4]}], "image_size": [10, 10]}
    class Tools:
        async def call_tool(self, name, arguments):
            calls.append(name)
            if name == "detector":
                return positions
            assert arguments["positions"] is positions
            return {"descriptions": [{"object_id": "car_1", "text": "A brown car."}]}
    namespace = {"fuse_enhanced_prompt": fuse_enhanced_prompt}
    source = expected_program(plan)
    validate_program(source, plan)
    exec(compile(source, "<validated test program>", "exec"), namespace)
    result = await namespace["enhance"](ctx, Tools())
    assert calls == ["detector", "semantic"]
    assert ctx["original_prompt"] in result
    assert "A brown car." in result
    assert "Spatial position description" not in result

