"""
src/model_setup.py

Loads the tokenizer (safe on CPU) and the 4-bit quantized Phi-2 model
(GPU only — run on Colab, not on a CPU-only laptop), then attaches LoRA
adapters via PEFT.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import MODEL_NAME, bnb_config, LORA_CONFIG


def load_tokenizer():
    """Loads Phi-2's tokenizer with left-padding, which is critical for
    correct batched generation with a causal (decoder-only) LM."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # Phi-2 has no dedicated pad token
    tokenizer.padding_side = "left"             # keep real tokens flush against the end of the sequence
    return tokenizer


def load_quantized_model():
    """
    Loads Phi-2 in 4-bit NF4 precision.
    GPU ONLY — bitsandbytes 4-bit loading requires CUDA. Run this in Colab.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Preps the quantized model so gradients can flow correctly during training
    # (casts some layers to fp32, enables checkpointing hooks, etc.)
    model = prepare_model_for_kbit_training(model)
    return model


def attach_lora(model):
    """Wraps the base model with trainable LoRA adapters. Everything else
    stays frozen — this is what makes fine-tuning 'parameter-efficient'."""
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check: should be well under 1% of total params
    return model


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_quantized_model()
    model = attach_lora(model)