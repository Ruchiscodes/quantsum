import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import MODEL_NAME, bnb_config, LORA_CONFIG

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # Phi-2 has no dedicated pad token, reuse EOS
    tokenizer.padding_side = "left"             # Left padding: critical for causal LM generation
    return tokenizer

def load_quantized_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)  # Preps quantized model for gradient flow
    return model

def attach_lora(model):
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # Sanity check: should be <1% of total params
    return model

if __name__ == "__main__":
    tok = load_tokenizer()
    model = load_quantized_model()
    model = attach_lora(model)