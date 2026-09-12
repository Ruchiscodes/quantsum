"""
src/benchmark_memory.py

Measures actual peak GPU memory usage for Phi-2 loaded in fp16 vs loaded
in 4-bit NF4 (QLoRA), and prints the real memory-reduction percentage.
This is the script that produces your honest "X% memory reduction" number
instead of you having to guess or assume one.

GPU REQUIRED — run this in Google Colab (T4), not on a CPU-only laptop.

Usage:
    python src/benchmark_memory.py
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.config import MODEL_NAME


def measure_memory(load_fn, label):
    """Loads a model via load_fn and reports the peak GPU memory used to do so."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_fn()

    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: peak GPU memory = {peak_gb:.2f} GB")

    del model
    torch.cuda.empty_cache()
    return peak_gb


def load_fp16():
    """Baseline: Phi-2 loaded in standard fp16 precision, no quantization."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )


def load_4bit():
    """Phi-2 loaded in 4-bit NF4 precision (the QLoRA quantization config)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. This benchmark measures GPU memory and must be "
            "run on a CUDA-enabled machine (e.g. Google Colab with a T4 GPU)."
        )

    fp16_mem = measure_memory(load_fp16, "FP16 baseline")
    int4_mem = measure_memory(load_4bit, "4-bit NF4 (QLoRA)")

    reduction = (fp16_mem - int4_mem) / fp16_mem * 100
    print(f"Memory reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()