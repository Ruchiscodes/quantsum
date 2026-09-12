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
# T4 (Colab's free GPU) is Turing architecture -> NO native bf16 tensor cores.
# Using bfloat16 on a T4 falls back to slow emulation and tanks throughput.
# A100/newer GPUs DO support bf16 natively and should use it.
_COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NF4 buckets match the real weight distribution better than plain int4
    bnb_4bit_compute_dtype=_COMPUTE_DTYPE,  # fp16 on T4, bf16 on A100+ -- auto-detected
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
    per_device_train_batch_size=8,   # raised from 2 -- 4-bit Phi-2 + LoRA leaves plenty of T4 headroom
    gradient_accumulation_steps=2,   # effective batch size = 8 * 2 = 16 (same as before, fewer slow accumulation loops)
    gradient_checkpointing=True,     # recompute activations on backward pass instead of storing them -> saves memory
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),  # True on T4
    bf16=torch.cuda.is_bf16_supported(),      # True on A100+, False on T4
    logging_steps=25,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # memory-efficient optimizer, standard pairing with QLoRA
    report_to="none",                # skip wandb/tensorboard overhead unless you want it
)