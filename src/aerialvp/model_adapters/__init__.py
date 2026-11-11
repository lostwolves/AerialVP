"""Local pretrained vision models, isolated from the MCP control process."""
def run_model(name, config, arguments):
    if name == 'yoloworld':
        from .detection import yoloworld
        return yoloworld(config, arguments)
    if name == 'grounding_dino':
        from .detection import grounding_dino
        return grounding_dino(config, arguments)
    if name == 'dam':
        from .dam import describe
        return describe(config, arguments)
    if name in {'spaceom', 'llava15', 'llava_onevision'}:
        from .vlm import describe
        return describe(name, config, arguments)
    raise ValueError(f'Unknown local model: {name}')
