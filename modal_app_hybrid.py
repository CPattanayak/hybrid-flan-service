import modal

# -------------------------------
# Volumes and Image
# -------------------------------
vol_cls = modal.Volume.from_name("distilbert-intent-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "transformers[torch]",
        "torch",
        "datasets",
        "gliner"
    )
)

app = modal.App("hybrid-gliner-service")

# -------------------------------
# Train DistilBERT Classifier
# -------------------------------
@app.function(image=image, gpu=None, volumes={"/vol_cls": vol_cls}, timeout=3600)
def train_classifier():
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from datasets import load_dataset

    MODEL = "distilbert-base-uncased"
    BASE_PATH = "/vol_cls/model"

    tok = DistilBertTokenizerFast.from_pretrained(MODEL)
    dataset = load_dataset("json", data_files="/vol_cls/intent_dataset.json")["train"]

    def tokenize(batch):
        return tok(batch["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL, num_labels=3)

    args = TrainingArguments(
        output_dir=BASE_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        eval_strategy="no",
        save_strategy="epoch",
        logging_strategy="epoch"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(BASE_PATH)
    tok.save_pretrained(BASE_PATH)

# -------------------------------
# FastAPI Hybrid Service
# -------------------------------
@app.function(image=image, gpu="T4", volumes={"/vol_cls": vol_cls})
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from gliner import GLiNER
    import torch, re

    CLS_PATH = "/vol_cls/model"

    # Load classifier
    intent_tok = AutoTokenizer.from_pretrained(CLS_PATH)
    intent_model = AutoModelForSequenceClassification.from_pretrained(CLS_PATH)

    # Load GLiNER extractor (inference only)
    gliner_model = GLiNER.from_pretrained("urchade/gliner_base")

    # ---------------- Tools ----------------
    def weather_tool(location: str):
        return f"Weather service response for {location}"

    def calculator_tool(expression: str):
        try: return str(eval(expression))
        except Exception as e: return str(e)

    def faq_tool(query: str):
        return f"FAQ response for: {query}"

    # ---------------- Pipeline ----------------
    def classify_intent(query: str) -> str:
        inputs = intent_tok(query, return_tensors="pt")
        outputs = intent_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return ["weather","calculator","faq"][pred]

    def extract_fields(query: str, intent: str):
        print("RAW Intent OUTPUT:", intent)

        if intent == "weather":
            entities = gliner_model.predict_entities(query, labels=["location"])
            if entities:
                return {"intent": intent, "location": entities[0]["text"]}

        elif intent == "calculator":
            # Regex fallback for math expressions
            match = re.search(r"[0-9\+\-\*/\%\^]+", query.replace(" ", ""))
            if match:
                return {"intent": intent, "expression": match.group()}
            entities = gliner_model.predict_entities(query, labels=["expression"])
            if entities:
                return {"intent": intent, "expression": entities[0]["text"]}
            return {"intent": intent, "expression": "0"}

        elif intent == "faq":
            entities = gliner_model.predict_entities(query, labels=["query"])
            if entities:
                return {"intent": intent, "query": entities[0]["text"]}

        return {"intent": intent}

    def run_pipeline(query: str) -> str:
        intent = classify_intent(query)
        fields = extract_fields(query, intent)

        if intent == "weather":
            return weather_tool(fields.get("location", query))
        elif intent == "calculator":
            return calculator_tool(fields.get("expression", "0"))
        elif intent == "faq":
            return faq_tool(fields.get("query", query))
        return faq_tool(query)

    # ---------------- FastAPI ----------------
    app = FastAPI(title="Hybrid GLiNER API", description="DistilBERT classification + GLiNER extraction", version="1.0")

    class QueryRequest(BaseModel):
        query: str
    class QueryResponse(BaseModel):
        response: str

    @app.post("/chat", response_model=QueryResponse)
    def chat_endpoint(req: QueryRequest):
        result = run_pipeline(req.query)
        return QueryResponse(response=result)

    return app
