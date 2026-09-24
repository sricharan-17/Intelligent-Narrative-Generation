import os
import sys

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from training.config import (
    MODEL_ASSIGNMENTS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    RANDOM_SEED,
)


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


def main():
    print("=" * 70)
    print("MEMBER 2 - LoRA SMOKE TEST")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"LoRA dropout: {LORA_DROPOUT}")
    print(f"Target modules: {TARGET_MODULES}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    torch.manual_seed(RANDOM_SEED)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Model loaded.")

    print("\nApplying LoRA...")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("LoRA applied successfully.")

    print("\nTrainable parameter summary:")
    model.print_trainable_parameters()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters were created.")

    print("\nChecking LoRA modules...")

    found = set()

    for name, module in model.named_modules():
        for target in TARGET_MODULES:
            if name.endswith(f".{target}"):
                found.add(target)

    print("Found target modules:")
    for target in TARGET_MODULES:
        status = "OK" if target in found else "MISSING"
        print(f"  {target:12} {status}")

    missing = set(TARGET_MODULES) - found

    if missing:
        raise RuntimeError(
            f"Missing LoRA target modules: {sorted(missing)}"
        )

    print("\n" + "=" * 70)
    print("LO-RA ATTACHMENT SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()