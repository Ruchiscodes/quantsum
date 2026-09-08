import torch
from transformers import BitsAndBytesConfig

MODEL_NAME = "microsoft/phi-2"

# 4-bit NormalFloat quantization config — the core memory-saving trick
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4: better than plain int4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16 even though weights are stored in 4-bit
    bnb_4bit_use_double_quant=True,         # quantizes the quantization constants too -> extra savings
)

LORA_CONFIG = dict(
    r=16,                 # rank of the low-rank matrices — bigger r = more capacity, more memory
    lora_alpha=32,        # scaling factor for LoRA updates
    target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],  # Phi-2's attention + MLP layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN_CONFIG = dict(
    output_dir="results/phi2-qlora-samsum",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 2 * 8 = 16
    gradient_checkpointing=True,     # trades compute for memory — recomputes activations instead of storing them
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=25,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # memory-efficient optimizer, standard for QLoRA
)