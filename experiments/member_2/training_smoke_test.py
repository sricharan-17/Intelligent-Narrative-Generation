import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from training.config import (
    MODEL_ASSIGNMENTS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    RANDOM_SEED,
)
from training.prepare_training_data import build_prompt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "splits", "train.jsonl")

MODEL_NAME = MODEL_ASSIGNMENTS["member_2"]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Smoke-test-only limit.
# This does NOT modify the frozen dataset.
MAX_LENGTH = 512


def main():
    print("=" * 70)
    print("MEMBER 2 - ONE-STEP TRAINING SMOKE TEST")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)

    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"LoRA dropout: {LORA_DROPOUT}")
    print(f"Target modules: {TARGET_MODULES}")
    print(f"Maximum smoke-test length: {MAX_LENGTH}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # --------------------------------------------------
    # Load ONE example from the frozen training split
    # --------------------------------------------------

    print("\nLoading one example from frozen train.jsonl...")

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        example = json.loads(f.readline())

    print(f"Record ID: {example['record_id']}")
    print(f"Input type: {example['input_type']}")
    print(f"Target: {example['target']}")

    # --------------------------------------------------
    # Build canonical project prompt
    # --------------------------------------------------

    prompt = build_prompt(example)

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Build Qwen chat-formatted input
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": "You are an interactive fantasy game narrator.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    target_text = example["target"] + tokenizer.eos_token

    # --------------------------------------------------
    # Tokenize prompt and target separately
    # --------------------------------------------------

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )

    target_tokens = tokenizer(
        target_text,
        add_special_tokens=False,
    )

    prompt_ids = prompt_tokens["input_ids"]
    target_ids = target_tokens["input_ids"]

    # --------------------------------------------------
    # Smoke-test truncation
    #
    # IMPORTANT:
    # Keep the target tokens so that the loss has
    # actual target positions to calculate.
    # --------------------------------------------------

    if len(target_ids) >= MAX_LENGTH:
        target_ids = target_ids[:MAX_LENGTH]
        prompt_ids = []
    else:
        max_prompt_length = MAX_LENGTH - len(target_ids)

        # Keep the most recent portion of the prompt.
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = prompt_ids + target_ids

    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [labels],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(input_ids)

    target_token_count = (labels != -100).sum().item()

    print(f"Token count used: {input_ids.shape[1]}")
    print(f"Target tokens used: {target_token_count}")

    if target_token_count == 0:
        raise RuntimeError(
            "No target tokens remain after smoke-test truncation."
        )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    print("Model loaded.")

    # --------------------------------------------------
    # Apply LoRA
    # --------------------------------------------------

    print("\nApplying LoRA...")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.train()

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=2e-4,
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    print(f"Initial loss: {loss.item():.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Loss is not finite."
        )

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    print("Running backward pass...")

    loss.backward()

    # --------------------------------------------------
    # Optimizer step
    # --------------------------------------------------

    print("Running optimizer step...")

    optimizer.step()
    optimizer.zero_grad()

    print("\n" + "=" * 70)
    print("ONE-STEP TRAINING SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()