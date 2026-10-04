import os
from dotenv import load_dotenv
from gradio_client import Client
from transformers import AutoTokenizer

load_dotenv()

SPACE_ID = "sricharan007/intelligent-narrative-api"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Reproduce the exact project prompt
exec(open("test_responder_prompt.py", encoding="utf-8").read())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\n===== FINAL CHAT-FORMATTED PROMPT =====")
print(formatted_prompt)

token = os.getenv("HF_SPACE_TOKEN")
if not token:
    raise RuntimeError("HF_SPACE_TOKEN is not set.")

client = Client(SPACE_ID, token=token)

print("\n===== MODEL OUTPUT =====")

result = client.predict(
    prompt=formatted_prompt,
    max_new_tokens=32,
    temperature=0.3,
    api_name="/generate_narrative",
)

print(result)
