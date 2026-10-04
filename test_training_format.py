import json
import os
from dotenv import load_dotenv
from gradio_client import Client
from transformers import AutoTokenizer

load_dotenv()

SPACE_ID = "sricharan007/intelligent-narrative-api"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Load the exact first training record
with open("data/splits/train.jsonl", "r", encoding="utf-8") as f:
    example = json.loads(next(f))

print("Record:", example["record_id"])
print("Expected target:", example["target"])

# Reconstruct the training-style prompt
s = example["setting"]
ic = example["input_character"]
rc = example["responder_character"]
w = example["world_state"]

prompt = f"""TASK
Generate exactly one natural next response from the RESPONDER CHARACTER.
Do not respond as the INPUT CHARACTER.
Use the conversation history and current input to determine the next response.
Stay consistent with the RESPONDER CHARACTER's persona, role, and situation.
Return only the response itself.

INPUT TYPE INSTRUCTIONS
If the input type is speech, generate what the responder should say next.
If the input type is action, generate the responder's appropriate next response to that action.

SETTING
Name: {s["name"]}
Category: {s["category"]}
Description: {s["description"]}
Background: {s["background"]}

INPUT CHARACTER
Name: {ic["name"]}
Persona: {ic["persona"]}

RESPONDER CHARACTER
Name: {rc["name"]}
Persona: {rc["persona"]}

WORLD STATE
Context: {w["context"]}
Room objects: {w["room_objects"]}
Room agents: {w["room_agents"]}
Object descriptions: {w["object_descriptions"]}
Carrying: {w["carrying"]}
Wearing: {w["wearing"]}
Wielding: {w["wielding"]}
Available actions: {w["available_actions"]}

HISTORY
{example["history"]}

PLAYER INPUT
{example["input"]}

INPUT TYPE
{example["input_type"]}

RESPONSE
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "system",
        "content": "You are an interactive fantasy game narrator."
    },
    {
        "role": "user",
        "content": prompt
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n===== PROMPT SENT TO MODEL =====\n")
print(formatted_prompt)

token = os.getenv("HF_SPACE_TOKEN")
if not token:
    raise RuntimeError("HF_SPACE_TOKEN is not set.")

client = Client(SPACE_ID, token=token)

print("\n===== MODEL OUTPUT =====\n")

result = client.predict(
    prompt=formatted_prompt,
    max_new_tokens=32,
    temperature=0.3,
    api_name="/generate_narrative"
)

print(result)

print("\n===== EXPECTED TRAINING TARGET =====")
print(example["target"])
