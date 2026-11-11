# Model setup

All visual checkpoints are stored locally and referenced by `configs/default.json`. Relative paths resolve against the repository root. Model weights are downloaded separately and retain their upstream licenses.

## Sources

| Tool | Upstream source | Download name |
|---|---|---|
| YOLO-World-X | [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/) / [checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt) | `yoloworld` |
| CLIP ViT-B/32 | [OpenAI CLIP](https://github.com/openai/CLIP) | `clip` |
| DAM | [NVIDIA DAM-3B-Self-Contained](https://huggingface.co/nvidia/DAM-3B-Self-Contained) | `dam` |
| SAM | [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) | `sam` |
| SpaceOm | [remyxai/SpaceOm](https://huggingface.co/remyxai/SpaceOm) | `spaceom` |
| Grounding-DINO | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | `grounding_dino` |
| LLaVA-1.5 | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | `llava15` |
| LLaVA-OneVision | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | `llava_onevision` |

Download only the models used by your chosen configuration. The main pipeline needs `yoloworld clip dam sam spaceom`. All HF snapshots include model configuration, processor/tokenizer files, weights, and any model-specific Python files. Preserve the complete directory structure, including DAM's nested components.

```bash
python scripts/setup_models.py --models yoloworld clip dam sam spaceom
python scripts/setup_models.py --models grounding_dino llava15 llava_onevision
```

The downloader resolves HF branches to commit hashes and records them in each model directory's `aerialvp_source.json`. For a fixed HF version, download that model separately with `--revision COMMIT_OR_TAG`. The CLIP download is checked against the upstream SHA256.

`--destination path/to/models` changes the download root. Update the corresponding configuration paths when using a different destination.

## Execution and memory

The MCP servers manage tool discovery and calls. Each visual-model invocation runs in a worker subprocess using the configured Python interpreter; by default this is the same interpreter that launched AerialVP. Models are loaded on demand, and process exit releases GPU allocations.

SAM generates masks from detector boxes, then its model is released before DAM loads. DAM and SpaceOm use float16 by default. LLaVA-1.5 and LLaVA-OneVision use NF4 4-bit quantization through bitsandbytes. These quantization settings are deployment options rather than paper-specified defaults.

The main pipeline requires CUDA because the self-contained DAM interface creates CUDA image-token inputs. Use a separate CUDA-capable environment for model execution if needed by setting `models.<name>.python` to that environment's Python executable.

## Compatibility

- Transformers is pinned to 4.51.3 for the model interfaces used here.
- SentencePiece 0.2.0 and Protobuf <6 support the self-contained DAM tokenizer. DAM loads the custom Python model implementation included in its official checkpoint with `trust_remote_code=True`.
- OpenCV is constrained to <4.12 to retain NumPy 1.x compatibility.
- OpenAI CLIP is installed from the official Git repository. The similarly named PyPI package is not a substitute.
- Grounding-DINO uses the native Transformers implementation with custom kernels disabled.
- OneVision preserves its processor's Qwen2 chat template and any-resolution image handling. When its configuration disables sliding-window attention, the adapter clears the window value to preserve full attention under Transformers 4.51.3.
- All returned boxes use original-image pixel coordinates in `xyxy` order.

## Individual model checks

These commands perform real inference on your image after the required checkpoints have been downloaded:

```bash
python scripts/smoke_models.py --model yoloworld --image "data/uav_image.jpg" --objects car building
python scripts/smoke_models.py --model dam --image "data/uav_image.jpg" --positions "work/smoke_yoloworld.json"
python scripts/smoke_models.py --model spaceom --image "data/uav_image.jpg" --positions "work/smoke_yoloworld.json"
python scripts/smoke_models.py --model grounding_dino --image "data/uav_image.jpg" --objects car building
python scripts/smoke_models.py --model llava15 --image "data/uav_image.jpg" --positions "work/smoke_yoloworld.json" --description semantic
python scripts/smoke_models.py --model llava_onevision --image "data/uav_image.jpg" --positions "work/smoke_yoloworld.json" --description relation
```

Description checks use one candidate by default. Set `--limit 0` to process all supplied candidates. Outputs and model logs are written to `work/`, which is excluded from version control.
