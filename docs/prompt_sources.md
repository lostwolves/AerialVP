# Prompt sources

The templates in `src/aerialvp/prompts.py` follow *[Towards Accurate UAV Image Perception: Guiding Vision-Language Models with Stronger Task Prompts](https://arxiv.org/abs/2512.07302)*.

| Templates | Source | Implementation |
|---|---|---|
| `TYPE_GUIDANCE`, `OBJECT_GUIDANCE`, `PLAN_GUIDANCE` | Fig. 6(a–c) | Transcribed with normalized typography and whitespace. The final `XXX` placeholder receives the current task input. |
| `FUSION_TEMPLATE` | Section III-F.2 | Preserves the original instruction and adds the task type, relevant objects, and selected enhancement information. |
| `SEMANTIC_GUIDANCE`, `RELATION_GUIDANCE` | Fig. 7; Sections III-B.2 and III-F | Expanded from the abbreviated examples to specify executable visual-description tasks. |
| `ENGINE_*_SYSTEM` | Sections III-B.1, III-E, and III-F | Meta-prompts for orchestration, tool selection, and program generation. Their complete wording is not provided in the paper. |
| Model input wrappers | Model interfaces and region association | Associate descriptions with target regions, object identifiers, and coordinates. |

## Validation and regeneration

The runtime validates tool names, arguments, schemas, and dependencies before execution. Invalid instructions are returned to the Task Engine with explicit error feedback for regeneration. Successful results are appended to execution history. This implements the validation–regeneration loop in Eqs. (5)–(9).

The default limit of five validation attempts is configurable through `engine.max_validation_attempts`. It is an implementation setting, not a paper-specified parameter. Setting it to zero permits retries until a valid instruction is produced. Infrastructure failures, such as unavailable API services or failed model loading, are reported separately from instruction-validation errors.

This loop checks execution correctness; it does not evaluate the accuracy of visual descriptions or final answers.

## Program generation and fusion

The Task Engine produces a Python function. The runtime checks its abstract syntax tree against the validated tool plan and adds imports and a module entry point. Running the generated program invokes the selected MCP tools and applies the fusion template.

Only the enhancement dimensions selected by the task plan are fused into the final prompt. Detection can serve as a prerequisite for region descriptions without adding position information to the final prompt.

## Model input formatting

SpaceOm and LLaVA receive the original image with the current target marked by a blue rectangle, together with object identifiers and original-image coordinates. DAM receives region masks produced by SAM. These wrappers adapt the tool guides to the individual model interfaces.
