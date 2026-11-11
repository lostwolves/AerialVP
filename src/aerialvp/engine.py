"""GPT Task Engine: analyze -> select -> generate, with explicit validation feedback."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .config import load_config
from .gpt import GPTClient, parse_object
from .planning import DIMENSIONS, PLAN_SCHEMA, TYPE_NAMES, expected_program, validate_instruction, validate_plan, validate_program
from .prompts import ENGINE_ANALYSIS_SYSTEM, ENGINE_SELECTION_SYSTEM, ENGINE_PROGRAM_SYSTEM, SEMANTIC_GUIDANCE, RELATION_GUIDANCE
from .repository import ToolRepository


class TaskEngine:
    def __init__(self, config_path=None, client=None):
        self.config = load_config(config_path)
        self.client = client or GPTClient(self.config["api"], model=self.config["engine"]["model"])
        self.history = []
        self.trace = []

    def _save_trace(self):
        if getattr(self, "run_dir", None):
            (self.run_dir / "engine_trace.json").write_text(json.dumps(self.trace, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _validated_decision(self, stage, system, payload, validator, executor=None):
        attempts = int(self.config["engine"].get("max_validation_attempts", 5))
        errors = []
        count = 0
        while attempts <= 0 or count < attempts:
            count += 1
            request = {**payload, "execution_history": self.history, "validation_feedback": errors}
            raw = await asyncio.to_thread(self.client.chat, json.dumps(request, ensure_ascii=False), system_prompt=system)
            event = {"stage": stage, "attempt": count, "model": self.client.model, "response": raw}
            try:
                decision = parse_object(raw)
                validator(decision)
                result = await executor(decision) if executor else decision
            except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                # Include validation errors in E(X_k, e_k), keep H_k unchanged on failure.
                message = f"{type(exc).__name__}: {exc}"
                event.update(status="rejected", error=message)
                errors.append({"instruction": raw, "error": message})
                self.trace.append(event)
                self._save_trace()
                continue
            except Exception as exc:
                # jsonschema and MCP schema failures can be repaired; infrastructure failures cannot.
                import jsonschema
                if isinstance(exc, jsonschema.ValidationError):
                    message = f"Schema validation: {exc.message}"
                    event.update(status="rejected", error=message)
                    errors.append({"instruction": raw, "error": message})
                    self.trace.append(event)
                    self._save_trace()
                    continue
                event.update(status="execution_failed", error=str(exc))
                self.trace.append(event)
                self._save_trace()
                raise
            event.update(status="accepted", result=result)
            self.trace.append(event)
            self.history.append({"stage": stage, "instruction": decision, "output": result})
            self._save_trace()
            return result
        raise RuntimeError(f"{stage}: no valid instruction after {attempts} attempts. See engine_trace.json; invalid instructions were not executed.")

    async def prepare(self, original_prompt: str, image_path: str, run_dir: str,
                      positions: dict | None = None, repository=None) -> Path:
        image = Path(image_path).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if not original_prompt.strip():
            raise ValueError("The original task prompt is empty")
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history, self.trace = [], []
        if repository is None:
            async with ToolRepository(self.config["config_path"]) as repo:
                return await self._prepare(original_prompt, str(image), positions, repo)
        return await self._prepare(original_prompt, str(image), positions, repository)

    async def _prepare(self, original_prompt, image_path, positions, repo):
        tools = await repo.list_tools()
        analyses = {}
        analysis_names = {"task_type_analysis", "task_object_analysis", "task_plan_analysis"}
        while set(analyses) != analysis_names:
            allowed = analysis_names - set(analyses)
            if not {"task_type_analysis", "task_object_analysis"} <= set(analyses):
                allowed.discard("task_plan_analysis")

            def validate(decision):
                validate_instruction(decision, tools, allowed)
                args = decision["arguments"]
                if args["original_prompt"] != original_prompt:
                    raise ValueError("Preserve original_prompt verbatim")
                if decision["tool"] == "task_plan_analysis":
                    if args["task_type"] != analyses["task_type_analysis"]["task_type"] or args["task_objects"] != analyses["task_object_analysis"]["task_object"]:
                        raise ValueError("Planning must use the completed type and object analysis outputs")

            async def execute(decision):
                output = await repo.call_tool(decision["tool"], decision["arguments"])
                if decision["tool"] == "task_type_analysis" and output.get("task_type") not in TYPE_NAMES:
                    raise ValueError("Unrecognized task type returned by analysis tool")
                if decision["tool"] == "task_object_analysis" and not output.get("task_object"):
                    raise ValueError("Task object analysis returned no objects")
                if decision["tool"] == "task_plan_analysis":
                    dims = output.get("task_enhancement_plan")
                    if not isinstance(dims, list) or len(set(dims)) != len(dims) or any(dim not in DIMENSIONS for dim in dims):
                        raise ValueError("Invalid enhancement plan dimensions")
                analyses[decision["tool"]] = output
                return output

            await self._validated_decision("task_prompt_analysis", ENGINE_ANALYSIS_SYSTEM,
                {"original_prompt": original_prompt, "analysis_results": analyses,
                 "available_tools": [tool for tool in tools if tool["name"] in allowed],
                 "required_output": {"tool": "one available tool name", "arguments": "object complying with its inputSchema"}}, validate, execute)

        context = {"original_prompt": original_prompt, "image_path": image_path,
                   "task_type": analyses["task_type_analysis"]["task_type"],
                   "task_objects": analyses["task_object_analysis"]["task_object"],
                   "enhancement_plan": analyses["task_plan_analysis"]["task_enhancement_plan"],
                   "existing_positions": positions or {"objects": [], "image_size": [0, 0]},
                   "semantic_guidance": SEMANTIC_GUIDANCE, "relation_guidance": RELATION_GUIDANCE}
        selected = self.config["tool_selection"]
        plan = await self._validated_decision("tool_selection", ENGINE_SELECTION_SYSTEM,
            {"task_context": context, "available_tools": [tool for tool in tools if tool["name"] in selected.values()],
             "configured_implementations": selected, "required_output_schema": PLAN_SCHEMA,
             "reference_syntax": "Use {'$ref':'ctx.image_path'}, {'$ref':'ctx.task_objects'}, {'$ref':'ctx.existing_positions'}, {'$ref':'ctx.semantic_guidance'}, {'$ref':'ctx.relation_guidance'}, or {'$ref':'preceding_step_id'}. Do not invent literal coordinates or descriptions.",
             "rules": "Steps are in execution order. ids must be lowercase Python identifiers. Each output maps a requested enhancement dimension to the matching step id. Output exactly the requested dimensions. A detector can run as a prerequisite even if its positions are not fused. Use supplied nonempty positions when available. No unused steps."},
            lambda value: validate_plan(value, context, tools, selected))

        # The engine emits actual Python; AST validation checks that it implements C exactly.
        def validate_source(value):
            if set(value) != {"source"}:
                raise ValueError("Return exactly {'source': '<Python function>'}")
            validate_program(value["source"], plan)

        program = await self._validated_decision("enhanced_prompt_program", ENGINE_PROGRAM_SYSTEM,
            {"task_context": context, "validated_configuration": plan,
             "required_output": {"source": "one async def enhance(ctx, tools) function"},
             "executable_contract": expected_program(plan),
             "rules": "Emit the executable Python function according to the contract. Preserve statement order, step ids, argument order, references, and the final fuse_enhanced_prompt call. Whitespace and comments may vary. No imports or Markdown in source. Runtime wrapper is supplied by the host."}, validate_source)
        manifest = {"config_path": self.config["config_path"], "context": context, "plan": plan,
                    "engine_model": self.client.model, "analysis_model": self.config["analysis"]["model"],
                    "paper_models": {"engine": "gpt-4.1", "analysis": "gpt-4o"},
                    "model_configuration": self.config["models"]}
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        header = ("# Generated by the AerialVP Task Engine; validated against manifest.json.\n"
                  "from aerialvp.runtime import fuse_enhanced_prompt, run_generated\n\n")
        footer = '\nif __name__ == "__main__":\n    run_generated(__file__, enhance)\n'
        program_path = self.run_dir / "enhance_program.py"
        program_path.write_text(header + program["source"] + footer, encoding="utf-8")
        return program_path

