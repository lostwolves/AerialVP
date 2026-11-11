from pathlib import Path
import re
from .common import load_image

def _objects(boxes, scores, categories, image):
    counters, output = {}, []
    for box, score, category in zip(boxes, scores, categories):
        key = re.sub(r'[^a-zA-Z0-9_]+', '_', category.strip()).strip('_') or 'object'
        counters[key] = counters.get(key, 0) + 1
        x1, y1, x2, y2 = box
        box = [max(0., min(float(x1), image.width)), max(0., min(float(y1), image.height)),
               max(0., min(float(x2), image.width)), max(0., min(float(y2), image.height))]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        output.append({'id': f'{key}_{counters[key]}', 'category': category,
                       'bbox': [round(v, 2) for v in box], 'confidence': float(score)})
    return {'objects': output, 'image_size': list(image.size)}

def yoloworld(config, arguments):
    import clip
    from ultralytics import YOLOWorld
    image = load_image(arguments)
    categories = arguments['task_objects']
    if not categories or not all(isinstance(c, str) and c.strip() for c in categories):
        raise ValueError('task_objects must be a nonempty list of category names')
    model = YOLOWorld(str(Path(config['checkpoint']).resolve(strict=True)))
    device = config.get('device', 'cuda:0')
    checkpoint = config.get('clip_checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError('YOLO-World needs a local CLIP ViT-B-32 checkpoint')
    model.model.clip_model = clip.load(checkpoint, device=device)[0]
    model.set_classes(categories)
    result = model.predict(image, imgsz=config.get('imgsz', 1280), conf=config.get('confidence', 0.25),
        iou=config.get('iou', 0.5), max_det=config.get('max_det', 300), device=device, verbose=False)[0]
    boxes = result.boxes
    labels = [categories[int(i)] for i in boxes.cls.cpu().tolist()]
    return _objects(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), labels, image)

def grounding_dino(config, arguments):
    import torch
    from transformers import AutoProcessor, GroundingDinoConfig, GroundingDinoForObjectDetection
    image = load_image(arguments)
    categories = arguments['task_objects']
    if not categories or not all(isinstance(c, str) and c.strip() for c in categories):
        raise ValueError('task_objects must be a nonempty list of category names')
    checkpoint = str(Path(config['checkpoint']).resolve(strict=True))
    device = config.get('device', 'cuda:0')
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model_config = GroundingDinoConfig.from_pretrained(checkpoint, local_files_only=True)
    model_config.disable_custom_kernels = True
    model = GroundingDinoForObjectDetection.from_pretrained(checkpoint, config=model_config,
        local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    text = '. '.join(c.strip().rstrip('.').lower() for c in categories) + '.'
    inputs = processor(images=image, text=text, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(outputs, input_ids=inputs['input_ids'],
        threshold=config.get('confidence', 0.3), text_threshold=config.get('text_threshold', 0.25),
        target_sizes=[(image.height, image.width)])[0]
    return _objects(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['text_labels'], image)
