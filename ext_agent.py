import os, json, random, re
import modal
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# Modal Setup
# -----------------------------
app = modal.App("extractor-agent")
vol = modal.Volume.from_name("extractor-vol", create_if_missing=True)

# Common image with all deps
common_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi","uvicorn","transformers","datasets","torch","accelerate","pydantic","peft"
    )
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "google/flan-t5-small"
MODEL_DIR = "/vol/model"
DATASET_FILE = "/vol/dataset.jsonl"
NUM_ENTRIES = 10000   # larger dataset for stronger training

# -----------------------------
# Dataset Generator (aligned prompt + validated)
# -----------------------------
@app.function(image=common_image, volumes={"/vol": vol}, timeout=600)
def generate_dataset():
    CUSTOMER_IDS = [f"C{100+i}" for i in range(NUM_ENTRIES)]
    ORDER_IDS = [f"ORD{500+i}" for i in range(NUM_ENTRIES)]
    FIRST_NAMES = ["John","Alice","Bob","Sophia","David","Emma","Liam","Olivia","Noah","Ava"]
    LAST_NAMES = ["Smith","Doe","Johnson","Brown","Taylor","Williams","Miller","Davis","Garcia","Martinez"]
    DOMAINS = ["test.com","foo.com","bar.org","mail.net"]

    with open(DATASET_FILE, "w") as f:
        for i in range(NUM_ENTRIES):
            cid = random.choice(CUSTOMER_IDS)
            fname = random.choice(FIRST_NAMES)
            lname = random.choice(LAST_NAMES)
            email = f"{fname.lower()}.{lname.lower()}@{random.choice(DOMAINS)}"
            oid = random.choice(ORDER_IDS)
            text = (
                f"Text: Customer {cid} {fname} {lname} email {email} order {oid}\n"
                "Output (valid JSON object only, no text, no explanation, must start with { and end with }):"
            )
            output = {
                "customer_id": cid,
                "first_name": fname,
                "last_name": lname,
                "email": email,
                "order_id": oid
            }
            output_str = json.dumps(output)
            entry = {"text": text, "Output": output_str}
            try:
                json.loads(output_str)  # validate
                f.write(json.dumps(entry) + "\n")
            except Exception as e:
                print("Invalid entry skipped:", e)
    print(f"Generated {NUM_ENTRIES} entries → {DATASET_FILE}")

# -----------------------------
# Training Function (LoRA)
# -----------------------------
@app.function(image=common_image, gpu="T4", volumes={"/vol": vol}, timeout=3600)
def train_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Attach LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(base_model, lora_config)

    dataset = load_dataset("json", data_files=DATASET_FILE)["train"]

    def preprocess(batch):
        inputs = batch["text"]
        targets = batch["Output"]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True)
        labels = tokenizer(targets, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(preprocess, batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        save_strategy="epoch",
        predict_with_generate=True,
        logging_dir="/vol/logs"
    )

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

# -----------------------------
# FastAPI Service
# -----------------------------
web_app = FastAPI(
    title="Extractor Agent API",
    description="Schema-guided extractor using LoRA fine-tuned Flan-T5",
    version="1.0.0"
)

class Input(BaseModel):
    text: str

def safe_parse(result: str):
    if not result.strip():
        return {"raw_output": ""}
    try:
        return json.loads(result)
    except:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return {"raw_output": result}

@web_app.post("/extract", summary="Extract structured JSON from text")
def extract(inp: Input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.config.tie_word_embeddings = False

    prompt = (
        f"Text: {inp.text}\n"
        f"Output (valid JSON object only, no text, no explanation, must start with {{ and end with }}):"
    )
    print("PROMPT:")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        do_sample=False
        # forced_bos_token_id=tokenizer.convert_tokens_to_ids("{"),
        # eos_token_id=tokenizer.convert_tokens_to_ids("}")
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("MODEL OUTPUT:")
    print(result)
    result="{"+result+"}" 

    parsed = safe_parse(result)
    return {"extracted": parsed}

# -----------------------------
# Modal Deployment
# -----------------------------
@app.function(image=common_image, gpu="T4", volumes={"/vol": vol})
@modal.asgi_app()
def fastapi_app():
    return web_app
