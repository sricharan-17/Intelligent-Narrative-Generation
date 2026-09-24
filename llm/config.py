from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# During development, place the trained LoRA adapter here:
ADAPTER_PATH = PROJECT_ROOT / "models" / "smollm2_narrative_qlora_2k"

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 128

TEMPERATURE = 0.7
TOP_P = 0.9

LOAD_IN_4BIT = True
USE_DOUBLE_QUANT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = "float16"
