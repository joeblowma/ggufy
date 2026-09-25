#!/usr/bin/env python3
"""Generate the multimodal (vision tower) fixtures: micro Qwen3-VL and Qwen2.5-VL
checkpoints built with transformers, and the mmproj files llama.cpp's own
converter writes for them.

Run with the ComfyUI venv that provides transformers + torch + gguf:
    /home/qt/genai/comfyui/nvenv/bin/python gen_vision_fixtures.py

Outputs (into src/test_fixtures/vision/<name>/):
    model.safetensors          bf16 checkpoint, text tower plus vision tower
    config.json                as transformers saved it (plus a top-level
                               "architectures" for qwen25vl, which 4.57 omits)
    preprocessor_config.json   image mean/std for the clip keys
    mmproj-f16.gguf            convert_hf_to_gguf.py --mmproj --outtype f16

$LLAMA_CPP points at the llama.cpp checkout (default ~/genai/llama.cpp).
"""
import json
import os
import subprocess
import sys

import torch
from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLImageProcessor, Qwen3VLConfig,
                          Qwen3VLForConditionalGeneration)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "src", "test_fixtures", "vision")
LLAMA = os.environ.get("LLAMA_CPP", os.path.expanduser("~/genai/llama.cpp"))

TEXT = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, vocab_size=4, max_position_embeddings=64, rms_norm_eps=1e-6,
            tie_word_embeddings=True)


def perturb(m):
    # Fresh init leaves norms at 1 and biases at 0, which would hide a swap.
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in m.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)


def save(m, d, mean, std, patch):
    os.makedirs(d, exist_ok=True)
    m.to(torch.bfloat16).save_pretrained(d)
    for extra in ("generation_config.json",):
        p = os.path.join(d, extra)
        if os.path.exists(p):
            os.remove(p)
    cfg_path = os.path.join(d, "config.json")
    cfg = json.load(open(cfg_path))
    if "architectures" not in cfg:
        cfg = {"architectures": [type(m).__name__], **cfg}
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    Qwen2VLImageProcessor(image_mean=mean, image_std=std, patch_size=patch, merge_size=2,
                          temporal_patch_size=2).save_pretrained(d)
    subprocess.run([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), d, "--mmproj",
                    "--outtype", "f16", "--outfile", os.path.join(d, "mmproj-f16.gguf")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


torch.manual_seed(0)
m = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
    text_config=dict(TEXT, head_dim=8, rope_theta=5000000.0,
                     rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
    vision_config=dict(depth=3, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                       patch_size=2, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16,
                       deepstack_visual_indexes=[1]),
    tie_word_embeddings=True))
perturb(m)
save(m, os.path.join(OUT, "qwen3vl"), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 2)

torch.manual_seed(0)
m = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(
    **TEXT, rope_theta=1000000.0, rope_scaling={"type": "mrope", "mrope_section": [1, 2, 1]},
    # depth 3 at least: llama.cpp maps mm.{bid} only for bid < depth.
    vision_config=dict(depth=4, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                       patch_size=2, spatial_merge_size=2, temporal_patch_size=2, window_size=8,
                       fullatt_block_indexes=[1, 3])))
perturb(m)
save(m, os.path.join(OUT, "qwen25vl"), [0.48145466, 0.4578275, 0.40821073],
     [0.26862954, 0.26130258, 0.27577711], 2)
print("wrote", OUT)
