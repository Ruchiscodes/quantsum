from datasets import load_dataset

def load_samsum():
    """Loads SAMSum: ~16k messenger-style dialogues with human summaries."""
    ds = load_dataset("knkarthick/samsum")
    return ds["train"], ds["validation"], ds["test"]

def format_example(example):
    """Turns a dialogue+summary pair into an instruction-style prompt for Phi-2."""
    prompt = (
        "Instruct: Summarize the following dialogue.\n"
        f"Dialogue:\n{example['dialogue']}\n"
        "Output:"
    )
    return {"prompt": prompt, "target": example["summary"]}

if __name__ == "__main__":
    train, val, test = load_samsum()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(format_example(train[0]))