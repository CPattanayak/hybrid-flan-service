import modal
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from contextlib import asynccontextmanager
import os
from datasets import Dataset

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("sql-agent")
vol = modal.Volume.from_name("sql-agent-vol", create_if_missing=True)

MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "/vol/sql_agent_lora"

common_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "transformers>=4.39.0",
        "torch",
        "accelerate",
        "pydantic",
        "peft",
        "sqlalchemy",
        "psycopg2-binary",
        "datasets",
    )
)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
DB_URL = "postgresql://postgres:password@localhost:5432/sql_training_db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schema() -> str:
    """
    Load schema metadata from Postgres.
    Returns a clean string like:
        customers(id, name, email)
        orders(id, customer_id, amount, order_date)
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT table_name, columns FROM schema_metadata", engine)
    schema = "\n".join(
        f"{row.table_name}({row.columns})" for _, row in df.iterrows()
    )
    return schema


def build_prompt(user_request: str, schema: str) -> str:
    """
    Unified prompt template used at BOTH training and inference time.
    Clear section markers prevent the model from confusing schema text
    with the instruction or expected output.
    """
    return (
        "### Schema:\n"
        f"{schema}\n\n"
        "### Task: Write PostgreSQL SQL only. No explanation.\n\n"
        f"### Request: {user_request}\n\n"
        "### SQL:\n"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    image=common_image,
    volumes={"/vol": vol},
    gpu="T4",
    timeout=14400,
    # schedule=modal.Cron("0 2 * * *"),  # uncomment to run daily at 2 AM UTC
)
def train_sql_agent_from_db():
    # Step 1: Load training data
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT input_text AS input, target_sql AS target FROM training_data",
        engine,
    )
    print(f"📦 Loaded {len(df)} training rows from DB.")

    # Step 2: Load schema once — inject into every training prompt
    schema_str = load_schema()
    print(f"📋 Schema for training:\n{schema_str}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        # Build schema-aware prompts — identical format to inference
        prompted = [build_prompt(inp, schema_str) for inp in batch["input"]]
        model_inputs = tokenizer(prompted, max_length=256, truncation=True)
        labels = tokenizer(
            text_target=batch["target"], max_length=128, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(
        preprocess, batched=True, remove_columns=dataset.column_names
    )

    # Step 3: Base model + LoRA
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Step 4: Train
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        warmup_steps=50,
        fp16=False,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # Step 5: Persist to Modal volume
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    vol.commit()
    print("✅ LoRA adapter + tokenizer saved:", os.listdir(OUTPUT_DIR))


# ---------------------------------------------------------------------------
# FastAPI inference service
# ---------------------------------------------------------------------------

class Query(BaseModel):
    request: str


_tokenizer = None
_model = None
_schema = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    global _tokenizer, _model, _schema

    # Load adapter + tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    _model.eval()

    # Load and validate schema
    _schema = load_schema()
    print(f"📋 Schema loaded:\n{repr(_schema)}")

    # Startup sanity check — catches broken adapters before real traffic
    test_prompt = build_prompt("Show all customers", _schema)
    print(f"🔍 Test prompt sent to model:\n{test_prompt}")
    test_inputs = _tokenizer(
        test_prompt, return_tensors="pt", max_length=256, truncation=True
    )
    with torch.no_grad():
        test_out = _model.generate(**test_inputs, max_new_tokens=64)
    print(
        "🔍 Startup sanity check →",
        _tokenizer.decode(test_out[0], skip_special_tokens=True),
    )

    yield

    del _tokenizer, _model, _schema


web_app = FastAPI(
    title="SQL Agent",
    description="Schema-aware LoRA fine-tuned SQL generator",
    version="1.0",
    lifespan=lifespan,
)


@web_app.post("/to_sql")
def to_sql(query: Query):
    prompt = build_prompt(query.request, _schema)

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )

    sql = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Request: {query.request!r}  →  SQL: {sql!r}")
    return {"sql": sql}


# ---------------------------------------------------------------------------
# Modal ASGI entrypoint
# ---------------------------------------------------------------------------

@app.function(image=common_image, volumes={"/vol": vol})
@modal.asgi_app()
def fastapi_app():
    return web_app
