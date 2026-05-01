# Efficient VLM Inference via Visual Token Compression

课程项目框架：在 Vision-Language Model 推理阶段研究 visual token compression 对 GPU memory、latency、throughput 和回答质量的影响，重点观察 efficiency-accuracy trade-off。

默认模型是 `Qwen/Qwen2.5-VL-3B-Instruct`，fallback 是 `Qwen/Qwen2-VL-2B-Instruct`。代码优先保证 Google Colab A100 能跑通 benchmark；对 Qwen2.5-VL 的 `fixed` 方法已经接入真实 visual-token pipeline：在 visual encoder 输出 image embeddings 后、送入 LLM prefill 前做 fixed-ratio pruning，并同步压缩 `input_ids`、`inputs_embeds`、`attention_mask`、`position_ids`。

## Project Structure

```text
vlm_token_compression/
├── configs/
│   └── default.yaml
├── data/
├── src/
│   ├── model_loader.py
│   ├── compression/
│   │   ├── base.py
│   │   ├── fixed_ratio_pruning.py
│   │   ├── importance_pruning.py
│   │   └── token_merging.py
│   ├── benchmark.py
│   ├── metrics.py
│   ├── inference.py
│   ├── plot_results.py
│   └── utils.py
├── notebooks/
│   └── colab_demo.ipynb
├── results/
├── plot_results.py
├── run_benchmark.py
└── README.md
```

## Colab Setup

In Colab, select `Runtime -> Change runtime type -> A100 GPU`, then run:

```bash
# Colab normally ships with CUDA-matched torch. If torch is missing, install it first:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -U torchvision transformers accelerate qwen-vl-utils datasets pillow pandas matplotlib tqdm psutil pynvml pyyaml
```

GPU check:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

If Transformers raises `KeyError: qwen2_5_vl` or `KeyError: qwen2_vl`, install a newer Transformers build:

```bash
pip install -U git+https://github.com/huggingface/transformers accelerate
```

## Quick Start

From the project root:

```bash
python run_benchmark.py --quick
```

Fallback to the smaller model:

```bash
python run_benchmark.py --quick --model-id Qwen/Qwen2-VL-2B-Instruct --dtype fp16
```

Full default benchmark:

```bash
python run_benchmark.py
```

Custom benchmark:

```bash
python run_benchmark.py \
  --methods none,fixed,importance,merging \
  --ratios 1.0,0.75,0.5,0.25,0.1 \
  --resolutions low,medium,high \
  --num-images 1,2,4 \
  --samples 3 \
  --max-new-tokens 64
```

Single-image inference:

```bash
python -m src.inference \
  --image /path/to/image.jpg \
  --question "Describe this image." \
  --method fixed \
  --ratio 0.5 \
  --resolution medium
```

## Outputs

Raw benchmark results:

```text
results/benchmark_results.csv
```

Summary table:

```text
results/summary_results.csv
```

Plots:

```bash
python plot_results.py
```

This writes:

```text
results/latency_vs_retention_ratio.png
results/memory_vs_retention_ratio.png
results/quality_vs_retention_ratio.png
results/efficiency_accuracy_tradeoff.png
```

## Compression Baselines

- `none`: no compression baseline, except images are normalized to the chosen `low/medium/high` visual-token budget.
- `fixed`: fixed-ratio pruning. 对 Qwen2.5-VL 会使用真实内部 adapter；其他模型会退回 proxy image budget。
- `importance`: token-norm top-k pruning. Tensor function keeps highest L2-norm visual tokens.
- `merging`: cosine-similarity anchor clustering. Tensor function merges similar visual tokens.

For Colab stability, `importance` and `merging` still use `apply_proxy_image_budget: true`: ratio `0.5` means the image is resized to roughly half the visual-token budget before tokenization. For Qwen2.5-VL `fixed`, `enable_internal_hooks: true` makes the code preserve the selected `low/medium/high` image budget first, then prune real visual embeddings after `get_image_features(...)`.

## Real Qwen2.5-VL Hook

The insertion point follows the current Hugging Face Qwen2.5-VL forward path:

```python
image_embeds = model.get_image_features(pixel_values, image_grid_thw).pooler_output
# fixed-ratio pruning happens here
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
outputs = language_model(inputs_embeds=inputs_embeds, ...)
```

The implementation lives in:

```text
src/compression/qwen2_5_vl_fixed.py
```

Instead of a blind PyTorch hook, it prepares a compressed generation prefill batch. This matters because `generate()` must see the same shortened sequence length that the KV cache sees.

Recorded extra fields:

```text
original_visual_tokens
kept_visual_tokens
compression_applied_internal
original_seq_len
compressed_seq_len
```

## Metrics

Each inference records:

- `latency_ms`
- `peak_gpu_memory_mb`
- `throughput_tokens_per_second`
- `generated_answer`
- `compression_method`
- `retention_ratio`
- `input_resolution`
- `num_images`
- `number_of_visual_tokens`
- `success`
- `oom`
- `quality_score`

Toy quality uses keyword matching. Replace this later with VQA-v2, TextVQA, ChartQA, LLaVA-Bench, or GPT-based judging.

## Current Limitations

- Internal real-token pruning is currently implemented for Qwen2.5-VL `fixed` only.
- The visual encoder still processes the full image. This method reduces LLM prefill sequence length and KV cache usage, but does not yet reduce ViT compute.
- `importance` and `merging` tensor functions are implemented, but true Qwen2.5-VL internal integration still needs per-image selected-index bookkeeping and MRoPE-safe sequence rebuilds similar to the fixed adapter.
- For non-Qwen2.5 models, the framework falls back to input image-budget compression.

## Next Steps

1. Extend `src/compression/qwen2_5_vl_fixed.py` to importance pruning by selecting top-k token-norm indices per image.
2. Extend it to token merging by replacing selected image placeholders with merged embeddings.
3. Move pruning earlier into the vision transformer if you want to reduce ViT compute too.
4. Add VQA-v2, TextVQA, ChartQA, or LLaVA-Bench evaluation.

References:

- Hugging Face Qwen2.5-VL docs: https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
- Qwen2.5-VL-3B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Qwen2-VL-2B-Instruct model card: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
