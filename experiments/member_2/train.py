"""
OFFICIAL MEMBER 2 TRAINING ENGINE: Qwen/Qwen2.5-3B-Instruct Standard LoRA

Project:
    Intelligent Narrative Generation for Interactive Gaming Using Fine-Tuned LLMs
Member:
    Member 2
Model:
    Qwen/Qwen2.5-3B-Instruct
Method:
    Standard LoRA (PEFT), FP16 precision on CUDA

Preserves all canonical prompt semantics, input/responder character distinction,
response-only -100 loss masking, frozen dataset splits, and official hyperparameters.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

from training.config import (
    MODEL_ASSIGNMENTS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    NUM_EPOCHS,
    LEARNING_RATE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
)
from training.prepare_training_data import build_prompt


# -----------------------------------------------------------------------------
# Paths & Fixed Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "splits", "train.jsonl")
VALIDATION_FILE = os.path.join(PROJECT_ROOT, "data", "splits", "validation.jsonl")

MEMBER_NAME = "member_2"
OFFICIAL_MODEL_NAME = MODEL_ASSIGNMENTS[MEMBER_NAME]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "member_2", "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "member_2", "logs")
RECORD_FILE = os.path.join(PROJECT_ROOT, "experiments", "member_2", "training_record.md")


# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------
class LightNarrativeDataset(Dataset):
    """
    PyTorch Dataset for LIGHT interaction turns.
    Preserves exact distinction:
      - input_character: character performing the current interaction
      - responder_character: character generating the response
    Constructs prompt via canonical build_prompt() and formats with Qwen ChatML.
    """

    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, Any]] = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset split not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]

        # 1. Canonical prompt construction (preserving exact character roles)
        prompt = build_prompt(example)

        # 2. ChatML formatting
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

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target_text = example["target"] + self.tokenizer.eos_token

        # 3. Separate tokenization
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        target_tokens = self.tokenizer(target_text, add_special_tokens=False)

        prompt_ids = prompt_tokens["input_ids"]
        target_ids = target_tokens["input_ids"]

        # 4. Truncation strategy: preserve target tokens up to max_length
        if len(target_ids) >= self.max_length:
            target_ids = target_ids[:self.max_length]
            prompt_ids = []
        else:
            max_prompt_len = self.max_length - len(target_ids)
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + target_ids
        # 5. Response-only -100 masking: prompt is masked out completely
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# -----------------------------------------------------------------------------
# Dynamic Data Collator
# -----------------------------------------------------------------------------
class DynamicCausalLMCollator:
    """
    Pads input_ids, attention_mask, and labels to the maximum length in the micro-batch.
    Padding tokens for labels are set to -100 so they are ignored by the loss function.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_max_len = max(len(item["input_ids"]) for item in batch)

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = batch_max_len - len(input_ids)

            # Right padding
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_attention_masks.append([1] * len(input_ids) + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Validation Function
# -----------------------------------------------------------------------------
def evaluate(model, val_loader, device):
    """
    Computes average loss on the frozen validation set.
    """
    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if torch.isfinite(loss):
                total_val_loss += loss.item()
                val_batches += 1

    model.train()
    return total_val_loss / max(1, val_batches)


# -----------------------------------------------------------------------------
# Checkpoint Saving & Resume Helper
# -----------------------------------------------------------------------------
def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    val_loss: float,
    checkpoint_dir: str,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save PEFT adapter
    model.save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler state for resumption
    state = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
    print(f"Checkpoint saved to: {checkpoint_dir}")


# -----------------------------------------------------------------------------
# Readiness Verification Function
# -----------------------------------------------------------------------------
def test_readiness():
    """
    Tests dataset loading, tokenization, canonical prompt construction,
    -100 masking, dynamic collation, and config validity without launching
    full 3B model training.
    """
    print("=" * 70)
    print("MEMBER 2 TRAINING ENGINE READINESS VERIFICATION")
    print("=" * 70)

    print(f"Official Model:            {OFFICIAL_MODEL_NAME}")
    print(f"LoRA Rank (r):             {LORA_R}")
    print(f"LoRA Alpha (alpha):        {LORA_ALPHA}")
    print(f"LoRA Dropout:              {LORA_DROPOUT}")
    print(f"LoRA Target Modules:       {TARGET_MODULES}")
    print(f"Epochs:                    {NUM_EPOCHS}")
    print(f"Learning Rate:             {LEARNING_RATE}")
    print(f"Micro Batch Size:          {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"Gradient Accumulation:     {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Warmup Ratio:              {WARMUP_RATIO}")
    print(f"Weight Decay:              {WEIGHT_DECAY}")
    print(f"Max Sequence Length:       {MAX_SEQUENCE_LENGTH}")
    print(f"Random Seed:               {RANDOM_SEED}")
    print(f"Precision:                 FP16 (torch.float16)")
    print(f"Checkpoints Dir:           {CHECKPOINTS_DIR}")

    # Check dataset files
    print("\n[Check 1/5] Verifying frozen split files...")
    assert os.path.exists(TRAIN_FILE), f"Missing {TRAIN_FILE}"
    assert os.path.exists(VALIDATION_FILE), f"Missing {VALIDATION_FILE}"
    print(f"  Train split exists:      {TRAIN_FILE}")
    print(f"  Validation split exists: {VALIDATION_FILE}")

    # Tokenizer check
    print("\n[Check 2/5] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer vocab size:    {tokenizer.vocab_size:,}")
    print(f"  Pad token ID:            {tokenizer.pad_token_id}")
    print(f"  EOS token ID:            {tokenizer.eos_token_id}")

    # Dataset & prompt check
    print("\n[Check 3/5] Testing single-sample prompt and -100 masking...")
    dataset = LightNarrativeDataset(TRAIN_FILE, tokenizer, max_length=MAX_SEQUENCE_LENGTH)
    print(f"  Loaded dataset size:     {len(dataset):,} examples")

    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    assert len(input_ids) == len(labels), "Length mismatch between input_ids and labels"
    masked_count = sum(1 for label in labels if label == -100)
    target_count = sum(1 for label in labels if label != -100)

    print(f"  Total sequence tokens:   {len(input_ids)}")
    print(f"  Masked prompt tokens:    {masked_count}")
    print(f"  Active target tokens:    {target_count}")
    assert masked_count > 0, "Prompt tokens were not masked with -100!"
    assert target_count > 0, "No active target tokens found!"

    # Dynamic collation check
    print("\n[Check 4/5] Testing dynamic batch collation...")
    collator = DynamicCausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    mini_batch = [dataset[0], dataset[1]]
    collated = collator(mini_batch)

    print(f"  Collated input_ids shape:      {collated['input_ids'].shape}")
    print(f"  Collated attention_mask shape: {collated['attention_mask'].shape}")
    print(f"  Collated labels shape:        {collated['labels'].shape}")

    assert collated["input_ids"].shape == collated["labels"].shape
    assert collated["input_ids"].shape == collated["attention_mask"].shape

    # CUDA environment & VRAM constraint audit
    print("\n[Check 5/5] Auditing CUDA environment & VRAM requirements...")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Device:         {device_name}")
        print(f"  VRAM:           {vram_gb:.2f} GB")

        if vram_gb < 12.0:
            print("\n  [IMPORTANT HARDWARE AUDIT NOTICE]")
            print(f"  Detected GPU has {vram_gb:.2f} GB VRAM.")
            print("  Standard FP16 LoRA for Qwen2.5-3B (~3.09B params) requires at least 14-16 GB VRAM.")
            print("  Per project protocol: Quantization, QLoRA, and gradient checkpointing are NOT")
            print("  automatically enabled. Launching full training on this 4 GB GPU will result in OOM.")
            print("  This script is fully configured and ready for execution on a 16-24 GB GPU environment.")
    else:
        print("  WARNING: CUDA is not available on this host.")

    print("\n" + "=" * 70)
    print("MEMBER 2 READINESS TEST: PASSED")
    print("=" * 70)
    return True


# -----------------------------------------------------------------------------
# Main Training Routine
# -----------------------------------------------------------------------------
def train(force_gpu: bool = False):
    """
    Main training execution function.
    """
    print("=" * 70)
    print("STARTING OFFICIAL MEMBER 2 TRAINING RUN")
    print("=" * 70)

    # 1. Reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 2. CUDA Device Audit
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official training run.")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {device} ({device_name})")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    if total_vram_gb < 12.0 and not force_gpu:
        print("\n" + "!" * 70)
        print("HARDWARE SAFETY STOP:")
        print(f"Current GPU has only {total_vram_gb:.2f} GB VRAM.")
        print("Standard FP16 LoRA with Qwen2.5-3B requires ≥14-16 GB VRAM.")
        print("To protect local system stability and adhere to project protocol,")
        print("do NOT train standard FP16 Qwen2.5-3B on this 4 GB GPU.")
        print("Run on a 16-24 GB GPU instance (e.g., cloud/Colab), or pass --force-gpu")
        print("if hardware conditions have changed.")
        print("!" * 70)
        sys.exit(1)

    # 3. Setup Dirs
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 4. Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Datasets & Loaders
    print("\nLoading frozen dataset splits...")
    train_dataset = LightNarrativeDataset(TRAIN_FILE, tokenizer, max_length=MAX_SEQUENCE_LENGTH)
    val_dataset = LightNarrativeDataset(VALIDATION_FILE, tokenizer, max_length=MAX_SEQUENCE_LENGTH)

    collator = DynamicCausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Train examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")

    # 6. Model & LoRA Setup
    print(f"\nLoading {OFFICIAL_MODEL_NAME} on CUDA in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        OFFICIAL_MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model.to(device)

    print("Applying PEFT LoRA...")
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
    model.print_trainable_parameters()

    # 7. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
    total_training_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(f"\nTotal Epochs:                  {NUM_EPOCHS}")
    print(f"Gradient Accumulation Steps:   {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Optimizer Steps Per Epoch:     {steps_per_epoch:,}")
    print(f"Total Optimizer Steps:         {total_training_steps:,}")
    print(f"Warmup Steps:                  {warmup_steps:,}")

    # 8. Training Loop
    start_time = time.time()
    global_step = 0
    best_val_loss = float("inf")

    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n" + "=" * 50)
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print("=" * 50)

        running_loss = 0.0
        accumulated_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Scale loss for gradient accumulation
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            accumulated_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_loss += accumulated_loss / GRADIENT_ACCUMULATION_STEPS
                accumulated_loss = 0.0

                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = running_loss / 100
                    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(
                        f"Epoch {epoch} | Step {global_step}/{total_training_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Peak VRAM: {peak_vram:.2f} GB"
                    )
                    running_loss = 0.0

        # Epoch Validation
        print(f"\nRunning validation at end of Epoch {epoch}...")
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        # Checkpoint
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch}")
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            checkpoint_dir=checkpoint_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINTS_DIR, "best")
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                checkpoint_dir=best_path,
            )

    total_time_sec = time.time() - start_time
    final_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("\n" + "=" * 70)
    print("MEMBER 2 OFFICIAL TRAINING COMPLETED")
    print(f"Total Time:      {total_time_sec / 3600:.2f} hours")
    print(f"Best Val Loss:   {best_val_loss:.4f}")
    print(f"Peak VRAM:       {final_peak_vram:.2f} GB")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Member 2 Qwen2.5-3B LoRA Training Engine")
    parser.add_argument(
        "--readiness-check",
        action="store_true",
        help="Run code, config, tokenization, and dynamic collator readiness test without full training.",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Bypass the 12 GB minimum VRAM hardware safety stop.",
    )

    args = parser.parse_args()

    if args.readiness_check:
        test_readiness()
    else:
        # If run directly without flags, test readiness first
        print("Verifying readiness before executing...")
        test_readiness()
        # If user explicitly wants to train:
        train(force_gpu=args.force_gpu)
