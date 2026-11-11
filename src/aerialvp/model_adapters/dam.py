import gc
from .common import load_image, objects_for, answer_text

def describe(config, arguments):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, SamModel, SamProcessor
    image = load_image(arguments)
    objects = objects_for(arguments, image)
    if not objects:
        return {'descriptions': []}
    sam_device = config.get('sam_device', 'cpu')
    sam_dtype = torch.float32 if sam_device == 'cpu' else torch.float16
    sam = SamModel.from_pretrained(config['sam_checkpoint'], local_files_only=True, torch_dtype=sam_dtype).to(sam_device).eval()
    processor = SamProcessor.from_pretrained(config['sam_checkpoint'], local_files_only=True)
    masks = []
    for obj in objects:
        inputs = processor(image, input_boxes=[[obj['bbox']]], return_tensors='pt').to(sam_device)
        if sam_dtype == torch.float16:
            inputs['pixel_values'] = inputs['pixel_values'].to(sam_dtype)
        with torch.inference_mode():
            output = sam(**inputs)
        candidates = processor.image_processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0][0]
        selected = candidates[int(output.iou_scores[0, 0].argmax())].numpy()
        masks.append(Image.fromarray((selected.astype(np.uint8) * 255)))
    del sam, processor, inputs, output, candidates
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = config.get('device', 'cuda:0')
    if not device.startswith('cuda'):
        raise ValueError('This self-contained DAM release requires CUDA for image token inputs')
    torch.cuda.set_device(device)
    model = AutoModel.from_pretrained(config['checkpoint'], trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation=config.get('attn_implementation', 'eager')).to(device).eval()
    dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
    guide = arguments['guide_prompt']
    query = guide if '<image>' in guide else '<image>\n' + guide
    descriptions = []
    for obj, mask in zip(objects, masks):
        with torch.inference_mode():
            result = dam.get_description(image, mask, query, streaming=False,
                temperature=config.get('temperature', 0.2), top_p=config.get('top_p', 0.5), num_beams=1,
                max_new_tokens=config.get('max_new_tokens', 512))
        descriptions.append({'object_id': obj['id'], 'text': answer_text(result)})
    return {'descriptions': descriptions}
