import os, json, random, re, torch
import modal
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

# -----------------------------
# Modal Setup
# -----------------------------
app = modal.App("qwen-extractor")
vol = modal.Volume.from_name("qwen-vol", create_if_missing=True)

common_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "transformers>=4.39.0",
        "datasets",
        "torch",
        "accelerate",
        "pydantic",
        "trl",       # pin stable TRL
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "tiktoken"
    )
)

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "/vol/model"
DATASET_FILE = "/vol/dataset.jsonl"
NUM_ENTRIES = 10   # ~50k samples

# -----------------------------
# Dataset Generator
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

            # Single variant per entry
            text_variant = f"Customer {cid} {fname} {lname} email {email} order {oid}"
            output = {
                "customer_id": cid,
                "first_name": fname,
                "last_name": lname,
                "email": email,
                "order": oid
            }

            entry = {"text": text_variant, "Output": json.dumps(output)}
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {NUM_ENTRIES} entries → {DATASET_FILE}")


# -----------------------------
# Training Function (QLoRA + Qwen2.5-3B-Instruct + SFTTrainer)
# -----------------------------
@app.function(image=common_image, gpu="A10G", volumes={"/vol": vol}, timeout=14400)
def train_model():
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    #-----------------------------------------------
    #Convert text into tokens.
    # Example:John Smith becomes [1234, 5678]
    # -----------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # ---------------------------------------------
    #  Without quantization:
    #  3B Model ≈ 12GB+
    # With 4-bit quantization:
    #  3B Model ≈ 3GB
    # ---------------------------------------------

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    #----------------------------------------------
    # Loads original Qwen model.
    # Think: 
    # Pretrained Brain

    # Already knows:

    # English
    # Reasoning
    # Coding
    # Instructions

    # We only teach:

    # Field Extraction
    # ----------------------------------------------
    base_model = prepare_model_for_kbit_training(base_model)
    # ------------------------------------------------------
    #  Makes quantized model trainable.

    # Without this:

    # Gradient issues
    # Training instability
    # ------------------------------------------------------

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear"
    )
    # -----------------------------------------------------
    # .

    # Traditional Fine Tuning:

    # Train all 3 Billion parameters

    # Huge cost.

    # LoRA:

    # Freeze original model
    # Train only small adapters

    # Example:

    # 3 Billion Parameters

    # Train only
    # 5-20 Million Parameters

    # Benefits:

    # Cheap
    # Fast
    # Small model files-
    # -----------------------------------------------------
    model = get_peft_model(base_model, lora_config)
    #  Adds trainable adapters.

    # Architecture:

    # Original Model
    #      +
    # LoRA Layers

    dataset = load_dataset("json", data_files=DATASET_FILE)["train"]

    def format_example(example):
        messages = [
            {
                "role": "system",
                "content": "Extract fields and return ONLY a valid JSON object. Do not explain.Use separate keys 'first_name' and 'last_name'. Do not combine them into 'name'."
            },
            {
                "role": "user",
                "content": example["text"]
            },
            {
                "role": "assistant",
                "content": example["Output"]
            }
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        #         Creates official Qwen format.

        # Example:

        # <|system|>
        # Extract fields

        # <|user|>
        # Customer C100...

        # <|assistant|>
        # {
        #  ...
        # }
        return {"text": text + tokenizer.eos_token}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=8,   # larger batch
        gradient_accumulation_steps=2,   # fewer steps
        num_train_epochs=2,              # fewer epochs
        learning_rate=2e-4,
        bf16=True,
        logging_steps=100,               # less logging
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_32bit"
    )


    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args
)

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

# -----------------------------
# FastAPI Service
# -----------------------------
web_app = FastAPI(title="Qwen Extractor API")

class Input(BaseModel):
    text: str

def safe_parse(result: str):
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return {"raw_output": match.group(0)}
    return {"raw_output": result}

tokenizer = None
model = None

@app.function(image=common_image, gpu="A10G", volumes={"/vol": vol}, timeout=3600)
@modal.asgi_app()
def fastapi_app():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    return web_app

@web_app.post("/extract")
def extract(inp: Input):
    messages = [
        {
            "role": "system",
            "content": "Return ONLY a valid JSON object. No explanations. No markdown."
        },
        {
            "role": "user",
            "content": inp.text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    parsed = safe_parse(result)
    print(result)
    return {"extracted": parsed}