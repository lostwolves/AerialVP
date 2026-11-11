from pathlib import Path
import math
import re

def load_image(arguments):
    from PIL import Image
    path = Path(arguments['image_path']).expanduser().resolve(strict=True)
    with Image.open(path) as source:
        return source.convert('RGB')

def objects_for(arguments, image):
    positions = arguments['positions']
    if positions.get('image_size') and list(positions['image_size']) != list(image.size):
        raise ValueError('Position coordinates refer to a different image size')
    objects = positions.get('objects', [])
    seen = set()
    for obj in objects:
        if not isinstance(obj.get('id'), str) or obj['id'] in seen:
            raise ValueError('Every position needs a unique string id')
        seen.add(obj['id'])
        box = obj.get('bbox', [])
        if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
            raise ValueError(f"Invalid bounding box for {obj['id']}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError(f"Bounding box outside the image for {obj['id']}: {box}")
    return objects

def marked_image(image, obj):
    from PIL import ImageDraw
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width = max(2, round(min(image.size) / 300))
    box = tuple(round(v) for v in obj['bbox'])
    draw.rectangle(box, outline='blue', width=width)
    draw.text((box[0] + width, box[1] + width), obj['id'], fill='blue')
    return marked

def object_question(guide, obj, objects, image):
    import json
    return (guide + '\n\nThe blue rectangle identifies the current target. '
        'The rectangle and its label are annotations, not target appearance. '
        f'Image dimensions: {image.width} x {image.height} pixels. '
        'All bounding boxes are [x1, y1, x2, y2] in original image pixels.\n'
        + 'Current target: ' + json.dumps(obj, ensure_ascii=False)
        + '\nOther task-relevant candidates: ' + json.dumps([o for o in objects if o['id'] != obj['id']], ensure_ascii=False)
        + '\nReturn the requested visual description of this target, not a final task answer.')

def answer_text(raw):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw, flags=re.S)
    if match:
        result = match.group(1).strip()
    else:
        result = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
        if '<think>' in result or '<answer>' in result:
            raise RuntimeError('Model output was truncated before a complete answer; increase max_new_tokens')
    if not result:
        raise RuntimeError('Model returned no visual description')
    return result

def model_kwargs(config):
    import torch
    dtype = getattr(torch, config.get('dtype', 'float16'))
    device = config.get('device', 'cuda:0')
    result = {'torch_dtype': dtype, 'local_files_only': True,
              'attn_implementation': config.get('attn_implementation', 'sdpa')}
    if config.get('load_in_4bit', False):
        from transformers import BitsAndBytesConfig
        result['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        result['device_map'] = {'': device}
    else:
        result['device_map'] = config.get('device_map', {'': device})
    if config.get('max_memory'):
        result['max_memory'] = {int(k) if str(k).isdigit() else k: v for k, v in config['max_memory'].items()}
    return result
