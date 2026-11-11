from .common import load_image, objects_for, marked_image, object_question, answer_text, model_kwargs

# Model-family chat wrapper, not a claimed verbatim paper prompt.
SPACEOM_SYSTEM = ('You are VL-Thinking, a helpful assistant with excellent reasoning ability. '
    'You should first think about the reasoning process and then provide the answer. '
    'Use <think>...</think> and <answer>...</answer> tags.')

def describe(name, config, arguments):
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration, LlavaOnevisionForConditionalGeneration
    image = load_image(arguments)
    objects = objects_for(arguments, image)
    if not objects:
        return {'descriptions': []}
    classes = {'spaceom': Qwen2_5_VLForConditionalGeneration, 'llava15': LlavaForConditionalGeneration,
               'llava_onevision': LlavaOnevisionForConditionalGeneration}
    loading_options = model_kwargs(config)
    if name == 'llava_onevision':
        hf_config = AutoConfig.from_pretrained(config['checkpoint'], local_files_only=True)
        # Transformers 4.51.3 Qwen2's causal mask checks sliding_window without
        # checking use_sliding_window. Honor this checkpoint's full attention.
        if not hf_config.text_config.use_sliding_window:
            hf_config.text_config.sliding_window = None
        loading_options['config'] = hf_config
    model = classes[name].from_pretrained(config['checkpoint'], **loading_options).eval()
    processor_kwargs = {'local_files_only': True}
    if name == 'spaceom':
        processor_kwargs.update(min_pixels=config.get('min_pixels', 256 * 28 * 28), max_pixels=config.get('max_pixels', 512 * 28 * 28))
    processor = AutoProcessor.from_pretrained(config['checkpoint'], **processor_kwargs)
    processor.tokenizer.padding_side = 'left'
    descriptions = []
    for obj in objects:
        marked = marked_image(image, obj)
        question = object_question(arguments['guide_prompt'], obj, objects, image)
        if name == 'spaceom':
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SPACEOM_SYSTEM}]},
                        {'role': 'user', 'content': [{'type': 'image', 'image': marked}, {'type': 'text', 'text': question}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif name == 'llava15':
            text = 'USER: <image>\n' + question + '\nASSISTANT:'
        else:
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[marked], return_tensors='pt').to(model.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=config.get('max_new_tokens', 1536), do_sample=False, use_cache=True)
        generated = output[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        descriptions.append({'object_id': obj['id'], 'text': answer_text(result)})
    return {'descriptions': descriptions}
