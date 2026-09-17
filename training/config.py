# ============================================================
# COMMON TRAINING CONFIGURATION
# Intelligent Narrative Generation for Interactive Gaming
# ============================================================

# -------------------------
# Dataset
# -------------------------

TRAIN_FILE = "data/splits/train.jsonl"
VALIDATION_FILE = "data/splits/validation.jsonl"
TEST_FILE = "data/splits/test.jsonl"

RANDOM_SEED = 42


# -------------------------
# Training task
# -------------------------

TASK_NAME = "interactive_narrative_generation"

INPUT_TYPES = [
    "speech",
    "action",
]


# -------------------------
# Fine-tuning method
# -------------------------

FINETUNING_METHOD = "LoRA"
PEFT_METHOD = "PEFT"

# These values will be finalized after
# hardware/model compatibility testing.

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# -------------------------
# Training parameters
# -------------------------

NUM_EPOCHS = 3

LEARNING_RATE = 2e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WARMUP_RATIO = 0.05

WEIGHT_DECAY = 0.01


# -------------------------
# Sequence length
# -------------------------

MAX_SEQUENCE_LENGTH = 2048


# -------------------------
# Evaluation
# -------------------------

EVAL_GENERATION_MAX_NEW_TOKENS = 128

EVAL_TEMPERATURE = 0.7

EVAL_TOP_P = 0.9


# -------------------------
# Reproducibility
# -------------------------

USE_SEED = True


# -------------------------
# Model assignment
# -------------------------

# Each member changes ONLY their assigned model.

MODEL_ASSIGNMENTS = {
    "member_1": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "member_2": "Qwen/Qwen2.5-3B-Instruct",
    "member_3": "Qwen/Qwen2.5-7B-Instruct",
}


# -------------------------
# Output directories
# -------------------------

CHECKPOINT_DIR = "training/checkpoints"
MODEL_OUTPUT_DIR = "models/fine_tuned"
RESULTS_DIR = "evaluation/results"


# -------------------------
# Experiment identification
# -------------------------

PROJECT_NAME = (
    "Intelligent Narrative Generation "
    "for Interactive Gaming Using Fine-Tuned Large Language Models"
)