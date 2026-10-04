from __future__ import annotations

import os

from dotenv import load_dotenv
from gradio_client import Client
from transformers import AutoTokenizer

from game.context.models import NarrativeContext

from .config import (
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
)
from .prompt_builder import NarrativePromptBuilder


load_dotenv()


HF_SPACE_ID = "sricharan007/intelligent-narrative-api"


class NarrativeGenerator:
    """Generate narrative responses using the remote Hugging Face API."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.client = None
        self.tokenizer = None

        self.device = "remote-huggingface"
        self.using_quantization = False

        self.prompt_builder = NarrativePromptBuilder()

    def load(self) -> None:
        """Connect to the private Hugging Face Space."""

        token = os.getenv("HF_SPACE_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_SPACE_TOKEN was not found. "
                "Make sure it is set in the project's .env file."
            )

        self.client = Client(
            HF_SPACE_ID,
            token=token,
        )

        # The tokenizer is only used locally to convert the
        # project's NarrativeContext messages into the same
        # chat-formatted prompt used by the fine-tuned model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, context: NarrativeContext) -> str:
        """Build the chat-formatted prompt for the remote model."""

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

        if self.client is None or self.tokenizer is None:
            raise RuntimeError(
                "NarrativeGenerator is not loaded. Call load() first."
            )

        prompt = self.build_prompt(context)

        result = self.client.predict(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            api_name="/generate_narrative",
        )

        return str(result).strip()

    def status(self) -> dict[str, object]:
        """Return runtime information useful for diagnostics."""

        return {
            "model": self.model_name,
            "adapter": "remote Hugging Face Space",
            "loaded": self.client is not None,
            "device": self.device,
            "cuda_available": False,
            "using_4bit_quantization": False,
            "space": HF_SPACE_ID,
        }