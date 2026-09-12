"""
src/config.py

Central place for every setting used across the project:
- 4-bit NF4 quantization config (the "QLoRA" part)
- LoRA adapter config (the "PEFT" part)
- Training hyperparameters (the "gradient checkpointing / throughput" part)

Import from here everywhere else instead of re-declaring these values,
so a single change (e.g. batch size) propagates everywhere.
"""

import torch
from transformers import BitsAndBytesConfig

MODEL_NAME = "microsoft/phi-2"

# ---------------------------------------------------------------------------
# 4-bit NormalFloat (NF4) quantization config.
# This is what shrinks Phi-2's weights from 16-bit to ~4-bit storage,
# which is where the big memory savings claim comes from.
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NF4 buckets match the real weight distribution better than plain int4
    bnb_4bit_compute_dtype=torch.bfloat16,  # matmuls still happen in bf16 even though weights are stored in 4-bit
    bnb_4bit_use_double_quant=True,         # also quantizes the quantization constants -> a bit more savings
)

# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) config — only these small matrices get trained,
# the rest of Phi-2 stays frozen. This is the "parameter-efficient" part.
# ---------------------------------------------------------------------------
LORA_CONFIG = dict(
    r=16,                 # rank of the low-rank matrices; higher = more capacity, more memory
    lora_alpha=32,        # scaling factor applied to the LoRA update
    target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],  # Phi-2 attention + MLP layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# Training hyperparameters.
# gradient_checkpointing=True is what lets you double the effective batch
# size without running out of GPU memory.
# ---------------------------------------------------------------------------
TRAIN_CONFIG = dict(
    output_dir="results/phi2-qlora-samsum",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,        # Enable FP16 for T4 GPU hardware acceleration
    bf16=False,       # T4 does not support native BF16 compute
    logging_steps=25,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)