"""
src/train.py

Fine-tunes Phi-2 with LoRA adapters on the SAMSum dialogue-summarization
dataset, using 4-bit NF4 quantization and gradient checkpointing.

GPU REQUIRED — run this in Google Colab (T4), not on a CPU-only laptop.

Usage:
    python -m src.train
"""

import time
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.model_setup import load_tokenizer, load_quantized_model, attach_lora
from src.data_prep import load_samsum, format_example
from src.config import TRAIN_CONFIG


def tokenize_fn(tokenizer, example, max_len=512):
    """Builds the full prompt+target text, tokenizes it, and sets labels
    equal to input_ids so the model learns to predict the summary tokens
    (causal LM training)."""
    text = format_example(example)
    full = text["prompt"] + " " + text["target"] + tokenizer.eos_token
    tokens = tokenizer(full, truncation=True, max_length=max_len, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    tokenizer = load_tokenizer()
    model = load_quantized_model()
    model = attach_lora(model)

    # Recomputes activations during the backward pass instead of storing them
    # from the forward pass -> frees enough memory to roughly double the
    # batch size you can fit on the same GPU.
    model.gradient_checkpointing_enable()

    train_ds, val_ds, _ = load_samsum()
    train_ds = train_ds.map(
        lambda e: tokenize_fn(tokenizer, e), remove_columns=train_ds.column_names
    )
    val_ds = val_ds.map(
        lambda e: tokenize_fn(tokenizer, e), remove_columns=val_ds.column_names
    )

    args = TrainingArguments(**TRAIN_CONFIG)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    start = time.time()
    trainer.train()
    elapsed_min = (time.time() - start) / 60
    print(f"Training took {elapsed_min:.1f} minutes")

    # Save only the small LoRA adapter weights (a few hundred MB), not the
    # full 2.7B base model.
    model.save_pretrained("results/phi2-qlora-adapter")
    tokenizer.save_pretrained("results/phi2-qlora-adapter")
    print("Adapter saved to results/phi2-qlora-adapter")


if __name__ == "__main__":
    main()