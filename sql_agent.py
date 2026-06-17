import modal
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch

# --- Modal setup ---
app = modal.App("sql-agent")
vol = modal.Volume.from_name("sql-agent-vol", create_if_missing=True)

MODEL_NAME = "google/flan-t5-small"
DATASET_FILE = "/vol/sql_dataset.jsonl"
OUTPUT_DIR = "/vol/sql_agent_lora"

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
        "trl",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "tiktoken"
    )
)

# --- Step 1: Generate dataset ---
@app.function(image=common_image, volumes={"/vol": vol})
def generate_dataset():
    examples = []

    # --- CREATE (INSERT) ---
    for i in range(1, 1001):
        name = f"Customer{i}"
        email = f"customer{i}@example.com"
        examples.append({
            "input": f"Translate to SQL: Add a new customer named {name} with email {email}",
            "target": f"INSERT INTO customers (name, email) VALUES ('{name}', '{email}');"
        })

    # --- READ (SELECT) ---
    for i in range(1, 1001):
        examples.append({
            "input": f"Translate to SQL: Show all orders for customer {i}",
            "target": f"SELECT * FROM orders WHERE customer_id={i};"
        })

    # --- UPDATE ---
    for i in range(1, 1001):
        new_email = f"user{i}@newmail.com"
        examples.append({
            "input": f"Translate to SQL: Update customer {i}'s email to {new_email}",
            "target": f"UPDATE customers SET email='{new_email}' WHERE id={i};"
        })

    # --- DELETE ---
    for i in range(1, 1001):
        examples.append({
            "input": f"Translate to SQL: Delete order {i}",
            "target": f"DELETE FROM orders WHERE order_id={i};"
        })

    # Save to JSONL
    Dataset.from_list(examples).to_json(DATASET_FILE)
    print(f"✅ Dataset saved to volume with {len(examples)} examples")


# --- Step 2: LoRA fine-tuning ---
@app.function(image=common_image, volumes={"/vol": vol}, gpu="T4", timeout=14400)
def train_sql_agent():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_json(DATASET_FILE)

    # Preprocess using T5's native target tokenizer.
    # No fixed padding here — your examples are ~15-30 tokens, so padding every
    # sample out to 128 was ~4-8x more compute than needed. DataCollatorForSeq2Seq
    # below pads dynamically per-batch instead, which is the main speed win.
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=128,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target"],
            max_length=128,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Inspect one sample to verify labels
    print("DEBUG SAMPLE INPUT:", tokenizer.decode(tokenized[0]["input_ids"]))
    print("DEBUG SAMPLE LABEL:", tokenizer.decode([t for t in tokenized[0]["labels"] if t != -100]))

    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # --- Actually wrap the model with LoRA (this was previously imported but unused) ---
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # T5 attention projections
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,            # LoRA converges faster than full fine-tuning; 20 was overkill
        learning_rate=5e-4,
        per_device_train_batch_size=32,  # model is tiny; 8 was leaving the GPU mostly idle
        fp16=False,                    # CRITICAL: T5's attention scaling overflows under fp16,
                                        # producing NaN gradients on every step (this is why every
                                        # grad_norm in your log was 'nan' and loss stuck at exactly
                                        # 0.0 — GradScaler was skipping every optimizer step, so the
                                        # model never actually trained, hence the verbatim echo).
                                        # bf16 would also fix this but needs Ampere+ (T4 doesn't
                                        # support it). Model is tiny, fp32 is cheap here.
        save_strategy="no",            # we only need the final adapter (saved manually below),
                                        # not a checkpoint written to disk every epoch
        logging_strategy="steps",
        logging_steps=20,
        disable_tqdm=True,             # tqdm's \r overwrites were hiding the loss in your logs
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Print the actual loss curve so you can confirm it went down
    print("=== LOSS HISTORY ===")
    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(entry)

    # --- Sanity check on multiple examples BEFORE saving, in eval mode ---
    model.eval()
    test_prompts = [
        "Translate to SQL: Delete order 567",
        "Translate to SQL: Show all orders for customer 12",
        "Translate to SQL: Add a new customer named Customer3 with email customer3@example.com",
        "Translate to SQL: Update customer 9's email to user9@newmail.com",
    ]
    print("=== POST-TRAIN SANITY CHECK ===")
    with torch.no_grad():
        for prompt in test_prompts:
            sample = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**sample, max_new_tokens=64, num_beams=4)
            print(prompt, "->", tokenizer.decode(out[0], skip_special_tokens=True))

    # Save the LoRA adapter (and tokenizer) — this now matches what fastapi_app loads via PeftModel
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    vol.commit()
    print("✅ LoRA adapter saved to volume")
    print("📂 Saved files:", os.listdir(OUTPUT_DIR))


# --- Step 3: FastAPI inference service ---
class Query(BaseModel):
    request: str

from contextlib import asynccontextmanager

tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    # Load tokenizer + base model + LoRA adapter once at startup
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()
    yield
    del tokenizer
    del model

web_app = FastAPI(
    title="SQL Agent",
    description="LoRA fine-tuned SQL generator",
    version="1.0",
    lifespan=lifespan
)

@web_app.post("/to_sql")
def to_sql(query: Query):
    inputs = tokenizer(f"Translate to SQL: {query.request}", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=4
        )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"sql": sql}


@app.function(image=common_image, volumes={"/vol": vol})
@modal.asgi_app()
def fastapi_app():
    return web_app
