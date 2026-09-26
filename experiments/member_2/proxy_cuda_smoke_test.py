"""
CUDA PROXY SMOKE TEST — NOT THE QWEN2.5-3B EXPERIMENT

PURPOSE:
Technical verification of CUDA training mechanics (device placement,
FP16 loading, forward pass, loss calculation, backward pass, LoRA gradient
computation, and AdamW optimizer step) using a lightweight proxy model
(Qwen/Qwen2.5-0.5B-Instruct) on local RTX 3050 Laptop GPU (4.0 GB VRAM).

NOTE:
This test is isolated. It does NOT modify or replace the official
Qwen/Qwen2.5-3B-Instruct experiment, dataset, splits, or training configs.
"""

import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from training.config import (
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    RANDOM_SEED,
)
from training.prepare_training_data import build_prompt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "splits", "train.jsonl")

PROXY_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MAX_LENGTH = 512


def main():
    print("=" * 70)
    print("CUDA PROXY SMOKE TEST — NOT THE QWEN2.5-3B EXPERIMENT")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        raise RuntimeError("CUDA is not available. This test requires a working CUDA device.")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Active Device: {device} ({device_name})")
    print(f"Total Device VRAM: {total_vram_gb:.2f} GB")

    vram_start_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"Initial allocated VRAM: {vram_start_gb:.3f} GB")

    print(f"\nProxy Model: {PROXY_MODEL_NAME}")
    print(f"LoRA rank: {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"LoRA dropout: {LORA_DROPOUT}")
    print(f"Target modules: {TARGET_MODULES}")
    print(f"Maximum sequence length: {MAX_LENGTH}")
    print(f"Model precision: torch.float16")

    # --------------------------------------------------
    # 1. Load one example from frozen train split
    # --------------------------------------------------
    print("\n[Step 1/8] Loading one example from frozen train.jsonl...")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Train split not found at: {TRAIN_FILE}")

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        example = json.loads(f.readline())

    print(f"Record ID: {example['record_id']}")
    print(f"Input type: {example['input_type']}")
    print(f"Target: {example['target']}")

    # --------------------------------------------------
    # 2. Build canonical prompt & tokenize with -100 mask
    # --------------------------------------------------
    print("\n[Step 2/8] Building canonical prompt and tokenizing...")
    prompt = build_prompt(example)

    tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    target_tokens = tokenizer(target_text, add_special_tokens=False)

    prompt_ids = prompt_tokens["input_ids"]
    target_ids = target_tokens["input_ids"]

    # Truncate prompt to fit within MAX_LENGTH while retaining target tokens
    if len(target_ids) >= MAX_LENGTH:
        target_ids = target_ids[:MAX_LENGTH]
        prompt_ids = []
    else:
        max_prompt_length = MAX_LENGTH - len(target_ids)
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids_list = prompt_ids + target_ids
    labels_list = [-100] * len(prompt_ids) + target_ids

    # Place tensors directly onto CUDA
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    target_token_count = (labels != -100).sum().item()
    masked_prompt_count = (labels == -100).sum().item()

    print(f"Total sequence tokens: {input_ids.shape[1]}")
    print(f"Masked prompt tokens (-100): {masked_prompt_count}")
    print(f"Active target tokens: {target_token_count}")
    print(f"Input tensor device: {input_ids.device}")
    print(f"Labels tensor device: {labels.device}")

    if target_token_count == 0:
        raise RuntimeError("No target tokens remain after smoke-test truncation.")

    # --------------------------------------------------
    # 3. Load proxy model on CUDA in FP16
    # --------------------------------------------------
    print("\n[Step 3/8] Loading proxy model on CUDA in FP16...")
    torch.cuda.reset_peak_memory_stats(0)

    model = AutoModelForCausalLM.from_pretrained(
        PROXY_MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model.to(device)
    print("Proxy model loaded and moved to CUDA.")
    print(f"Model primary dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    vram_model_loaded_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"VRAM after model load: {vram_model_loaded_gb:.3f} GB")

    # --------------------------------------------------
    # 4. Attach LoRA to all 7 target modules
    # --------------------------------------------------
    print("\n[Step 4/8] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.train()

    print("LoRA attached successfully.")
    model.print_trainable_parameters()

    # Verify target module presence
    found_targets = set()
    for name, module in model.named_modules():
        for t in TARGET_MODULES:
            if name.endswith(f".{t}"):
                found_targets.add(t)

    print(f"Target modules attached: {sorted(list(found_targets))}")
    missing_targets = set(TARGET_MODULES) - found_targets
    if missing_targets:
        raise RuntimeError(f"Missing target modules in LoRA model: {missing_targets}")

    # --------------------------------------------------
    # 5. Initialize Optimizer
    # --------------------------------------------------
    print("\n[Step 5/8] Setting up AdamW optimizer for trainable LoRA parameters...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    # --------------------------------------------------
    # 6. Forward pass on CUDA & Loss verification
    # --------------------------------------------------
    print("\n[Step 6/8] Executing forward pass on CUDA...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    print(f"Loss computed: {loss.item():.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite!")

    vram_forward_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"VRAM after forward pass: {vram_forward_gb:.3f} GB")

    # --------------------------------------------------
    # 7. Backward pass on CUDA & Gradient verification
    # --------------------------------------------------
    print("\n[Step 7/8] Executing backward pass on CUDA...")
    loss.backward()
    print("Backward pass completed on CUDA.")

    # Verify that LoRA parameters received valid gradients
    grad_ok_count = 0
    total_lora_tensors = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_lora_tensors += 1
            if param.grad is not None and torch.isfinite(param.grad).all():
                grad_ok_count += 1
            else:
                print(f"WARNING: No valid gradient for trainable parameter: {name}")

    print(f"LoRA trainable tensors with valid finite gradients: {grad_ok_count}/{total_lora_tensors}")
    if grad_ok_count != total_lora_tensors or total_lora_tensors == 0:
        raise RuntimeError("LoRA gradient verification failed!")
    print("Gradient verification PASSED.")

    # --------------------------------------------------
    # 8. Optimizer step on CUDA
    # --------------------------------------------------
    print("\n[Step 8/8] Executing optimizer step on CUDA...")
    optimizer.step()
    optimizer.zero_grad()
    print("Optimizer step completed successfully.")

    # VRAM and peak statistics
    vram_final_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    print(f"\nFinal allocated VRAM: {vram_final_gb:.3f} GB")
    print(f"Peak VRAM during training step: {peak_vram_gb:.3f} GB (out of {total_vram_gb:.2f} GB total)")

    print("\n" + "=" * 70)
    print("CUDA PROXY SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
