from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from game.context.models import NarrativeContext

from .config import (
    ADAPTER_PATH,
    COMPUTE_DTYPE,
    LOAD_IN_4BIT,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    QUANT_TYPE,
    TEMPERATURE,
    TOP_P,
    USE_DOUBLE_QUANT,
)
from .prompt_builder import NarrativePromptBuilder


class NarrativeGenerator:
    """Generate narrative responses using the fine-tuned SmolLM2 adapter."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        adapter_path: Path | str = ADAPTER_PATH,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ):
        self.model_name = model_name
        self.adapter_path = Path(adapter_path)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = None
        self.model = None
        self.device = None
        self.using_quantization = False

        self.prompt_builder = NarrativePromptBuilder()

    def load(self) -> None:
        """Load the base model and trained LoRA adapter."""

        if not self.adapter_path.exists():
            raise FileNotFoundError(
                "Fine-tuned LoRA adapter was not found at:\n"
                f"{self.adapter_path}"
            )

        adapter_weights = self.adapter_path / "adapter_model.safetensors"

        if not adapter_weights.exists():
            raise FileNotFoundError(
                f"Adapter weights not found at:\n{adapter_weights}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self._load_gpu()
        else:
            self._load_cpu()

        self.model.eval()

    def _load_gpu(self) -> None:
        """Load using 4-bit bitsandbytes quantization on CUDA."""

        compute_dtype = (
            torch.float16
            if COMPUTE_DTYPE == "float16"
            else torch.bfloat16
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
        )

        self.device = next(self.model.parameters()).device
        self.using_quantization = True

    def _load_cpu(self) -> None:
        """Load the model without CUDA quantization for CPU inference."""

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
        )

        self.model = self.model.to("cpu")
        self.device = torch.device("cpu")
        self.using_quantization = False

    def build_prompt(self, context: NarrativeContext) -> str:
        """Build the exact chat-formatted prompt used for generation."""

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Call load() first."
            )

        messages = self.prompt_builder.build_messages(context)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, context: NarrativeContext) -> str:
        """Generate one narrative response from a NarrativeContext."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "NarrativeGenerator is not loaded. Call load() first."
            )

        prompt = self.build_prompt(context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return response

    def status(self) -> dict[str, object]:
        """Return runtime information useful for diagnostics."""

        return {
            "model": self.model_name,
            "adapter": str(self.adapter_path),
            "loaded": self.model is not None,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "using_4bit_quantization": self.using_quantization,
        }
