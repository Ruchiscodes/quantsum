"""
src/evaluate.py

Compares zero-shot Phi-2 vs the LoRA fine-tuned Phi-2 on ROUGE-1/ROUGE-L,
using the SAMSum test split. Produces the actual "% lift" number for the
project's headline metric.

GPU REQUIRED — run this in Google Colab (T4), after src/train.py has
produced results/phi2-qlora-adapter.

Usage:
    python -m src.evaluate
"""

import evaluate as hf_evaluate
from peft import PeftModel

from src.model_setup import load_tokenizer, load_quantized_model
from src.data_prep import load_samsum, format_example

rouge = hf_evaluate.load("rouge")


def generate_summary(model, tokenizer, dialogue, max_new_tokens=60):
    """Generates a summary for a single dialogue using greedy decoding."""
    prompt = format_example({"dialogue": dialogue, "summary": ""})["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Output:")[-1].strip()


def run_eval(model, tokenizer, test_ds, n=200):
    """Generates summaries for n test examples and computes ROUGE scores
    against the human reference summaries."""
    preds, refs = [], []
    for ex in test_ds.select(range(min(n, len(test_ds)))):
        preds.append(generate_summary(model, tokenizer, ex["dialogue"]))
        refs.append(ex["summary"])
    return rouge.compute(predictions=preds, references=refs)


def main():
    tokenizer = load_tokenizer()
    _, _, test_ds = load_samsum()

    print("== Zero-shot baseline ==")
    base_model = load_quantized_model()
    baseline_scores = run_eval(base_model, tokenizer, test_ds)
    print(baseline_scores)

    print("== Fine-tuned (LoRA adapter) ==")
    ft_model = PeftModel.from_pretrained(base_model, "results/phi2-qlora-adapter")
    ft_scores = run_eval(ft_model, tokenizer, test_ds)
    print(ft_scores)

    lift_r1 = (ft_scores["rouge1"] - baseline_scores["rouge1"]) / baseline_scores["rouge1"] * 100
    lift_rl = (ft_scores["rougeL"] - baseline_scores["rougeL"]) / baseline_scores["rougeL"] * 100

    print(f"ROUGE-1 lift: {lift_r1:.1f}%")
    print(f"ROUGE-L lift: {lift_rl:.1f}%")


if __name__ == "__main__":
    main()