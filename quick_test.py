import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# just use base model, no lora fine-tuning for speed
MODEL_PATH = "models/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto", local_files_only=True)
model.eval()

with open("data/sft_challenge_test.jsonl", "r", encoding="utf-8") as f:
    tests = [json.loads(line) for line in f]

all_output = []

for test in tests:
    prompt = (
        "You are a university IT support assistant.\n\n"
        "User: " + test["prompt"] + "\n\nAssistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    all_output.append("\n====================")
    all_output.append("QUESTION:")
    all_output.append(test["prompt"])
    all_output.append("\nEXPECTED POINTS:")
    all_output.append(str(test["expected_points"]))
    all_output.append("\nMODEL:")
    all_output.append(response)

# save v1
with open("SFT_V1_Test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_output))

print("done v1")
