"""Paper Fig. 6 transcriptions, fusion template, and disclosed reproduction prompts.

Fig. 7 truncates tool guides with ellipses; the complete guides below implement
the described tool behavior. Engine meta-prompts implement Eqs. (5)-(9),
not verbatim text from the paper. See docs/prompt_sources.md.
"""
import json

TYPE_GUIDANCE = '''You are an assistant responsible for analyzing the task type of a UAV's perception mission. I will provide you with a mission instruction, and you need to identify the task type from the following categories: 1. Visual Grounding: Locate a target from a descriptive statement in an image. 2. Visual Question Answering: Select the correct option for a question with multiple choices. 3. Visual Reasoning: Determine if a declarative statement is true or false.
Analyze the provided mission instruction and return only the task_type in a dictionary format. Example:
- Instruction: "Locate the green car next to the red house"
- Result: {"task_type": "Visual Grounding"}
Now, I will provide you with a mission instruction. Please return only the task type analysis result according to the above standards without additional explanations: XXX'''

OBJECT_GUIDANCE = '''You are an assistant responsible for analyzing the objects of interest in a UAV's perception mission. I will provide you with a mission instruction, and you need to identify and return both target objects and anchor objects (if present) in a single list. The target object is the primary focus of the task (always present), while the anchor object is a reference point for context or location (optional).
Analyze the provided mission instruction and return only the task_object in a dictionary format. Include all relevant objects, even if they are anchor objects. Example:
- Instruction: "Locate the green car next to the red house"
- Result: {"task_object": ["car", "house"]}
Now, I will provide you with a mission instruction. Please return only the objects of interest analysis result according to the above standards without additional explanations: XXX'''

PLAN_GUIDANCE = '''You are an assistant responsible for analyzing the required task enhancement work for a UAV's perception mission. I will provide you with a mission instruction, the task type, and the task objects. You need to determine the necessary enhancement tasks:
- For Visual Grounding: Return {"task_enhancement_plan": ["spatial_position_description", "semantic_description", "spatial_relationship_description"]}.
- For Visual Question Answering / Visual Reasoning: Return {"task_enhancement_plan": ["semantic_description", "spatial_relationship_description"]}.
If the instruction explicitly provides certain information (e.g., coordinate data), remove the corresponding task from the list. Analyze the provided mission instruction and return only the task_enhancement_plan in a dictionary format. Do not include any other details. Example:
- Instruction: "Locate the green car next to the red house"
- Task Type: "Visual Grounding"
- Task Objects: ["car", "house"]
- Result: {"task_enhancement_plan": ["spatial_position_description", "semantic_description", "spatial_relationship_description"]}
Now, I will provide you with a mission instruction, task type, and task objects. Please return only the task enhancement work analysis result according to the above standards without additional explanations: XXX'''

SEMANTIC_GUIDANCE = '''I will provide you with a UAV image, please provide the semantic description of the target in the image, including its category, color, appearance attributes, and surrounding scene context. Describe the indicated target with common, visually grounded concepts and distinguish it from nearby objects. Report observable details without answering the original perception task.'''
RELATION_GUIDANCE = '''I will provide you with a UAV image, please provide the spatial relationship description of the target in the image, including its spatial position, relative directions, relative distances, and neighborhood relations to surrounding objects. Use the supplied object identifiers and original-image coordinates to associate descriptions with the correct targets. Describe visible relationships without answering the original perception task.'''

FUSION_TEMPLATE = ('You are an assistant responsible for UAV Image {task_type} Task. '
                   'The target objects you need to pay attention to include: {task_object}. '
                   '{original_task_instruction}. Here is some additional information: {enhancement_information}')

ENGINE_ANALYSIS_SYSTEM = '''You are the LLM-based Task Engine of AerialVP. Your job is to manage task analysis using the registered MCP Prompt Analysis Tools. Use the original prompt, execution history, tool descriptions and stage requirements to decide the next executable tool instruction. Determine task type and target/anchor objects before requesting the enhancement plan. You do not answer the perception task. Preserve the original instruction. Follow the supplied output schema and tool interfaces exactly. Return only the requested JSON object. If validation feedback is provided, correct the reported error using the same task context.'''

ENGINE_SELECTION_SYSTEM = '''You are the LLM-based Task Engine of AerialVP in Tool Selection. Map the structured task representation to the configured MCP enhancement tools. Determine required arguments, invocation order and data dependencies. Only requested enhancement dimensions belong in the final output; prerequisite detection may still be needed by description tools. Use registered tools, supplied context references and preceding tool outputs; do not invent visual evidence or target coordinates. Return only the JSON execution configuration complying with the provided schema. Correct explicit validation errors before continuing.'''

ENGINE_PROGRAM_SYSTEM = '''You are the LLM-based Task Engine of AerialVP in Enhanced Prompt Generation. Generate an executable Python workflow implementing the already validated tool-execution configuration. The generated function must invoke the selected MCP tools, pass their outputs along the specified dependencies, and fuse only selected enhancement outputs with the unchanged original task prompt. The execution contract and Python interface are supplied by the host. Return only the specified JSON object containing the Python source. Use validation feedback to regenerate invalid code. Do not produce the final perception answer or fabricate tool outputs.'''


def render_guidance(template, payload):
    if not template.endswith("XXX"):
        raise ValueError("Paper analysis template must end with its XXX input placeholder")
    return template[:-3] + payload


def build_type_prompt(original_prompt):
    return render_guidance(TYPE_GUIDANCE, original_prompt)


def build_object_prompt(original_prompt):
    return render_guidance(OBJECT_GUIDANCE, original_prompt)


def build_plan_prompt(original_prompt, task_type, task_objects):
    payload = f'\nInstruction: {original_prompt}\nTask Type: {task_type}\nTask Objects: {json.dumps(task_objects, ensure_ascii=False)}'
    return render_guidance(PLAN_GUIDANCE, payload)


def fuse_prompt(original_prompt, task_type, task_objects, enhancement_information):
    return FUSION_TEMPLATE.format(task_type=task_type, task_object=", ".join(task_objects),
                                 original_task_instruction=original_prompt,
                                 enhancement_information=enhancement_information)
