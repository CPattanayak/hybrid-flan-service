# Hybrid AI Training Platform

A practical implementation of multiple AI model training architectures using Hugging Face, LoRA, QLoRA, Modal, and FastAPI.

This repository demonstrates how different transformer architectures are trained and deployed for real-world applications such as Intent Classification, Text-to-SQL Generation, and Information Extraction.

---

## Architecture Overview

![Architecture Diagram](docs/images/architecture.png)

| Architecture | Model | Use Case |
|-------------|---------|----------|
| Encoder | DistilBERT | Intent Classification |
| Encoder-Decoder | FLAN-T5 | Text-to-SQL Generation |
| Decoder Only | Qwen 2.5 | Information Extraction |

---

## Encoder Model - Intent Classification

### Model
- DistilBERT
- Hugging Face Trainer

### Training Flow

```text
Input Text
    ↓
Tokenization
    ↓
DistilBERT Encoder
    ↓
Classification Head
    ↓
Intent Prediction
```

### Source Code

- [modal_app_hybrid.py](https://github.com/CPattanayak/hybrid-flan-service/blob/main/modal_app_hybrid.py)

---

## Encoder-Decoder Model - Text to SQL

### Model
- FLAN-T5
- LoRA Fine Tuning
- Seq2SeqTrainer

### Training Flow

```text
Natural Language
        ↓
     Encoder
        ↓
Context Representation
        ↓
     Decoder
        ↓
     SQL Query
```

### Source Code

- [ext_agent.py](https://github.com/CPattanayak/hybrid-flan-service/blob/main/ext_agent.py)

---

## Decoder Only Model - Information Extraction

### Model
- Qwen 2.5 3B Instruct
- QLoRA
- SFTTrainer

### Training Flow

```text
Prompt
   ↓
Qwen Decoder
   ↓
LoRA Adapters
   ↓
Generated JSON
```

### Source Code

- [sql_agent.py](https://github.com/CPattanayak/hybrid-flan-service/blob/main/sql_agent.py)

---

## Technology Stack

### AI / ML

- Hugging Face Transformers
- Datasets
- PEFT
- LoRA
- QLoRA
- TRL
- BitsAndBytes

### Deployment

- Modal
- FastAPI
- Uvicorn

### Language

- Python

---

## Project Structure

```text
hybrid-flan-service/
│
├── modal_app_hybrid.py
├── ext_agent.py
├── sql_agent.py
├── docs/
│   └── images/
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/CPattanayak/hybrid-flan-service.git

cd hybrid-flan-service

pip install -r requirements.txt
```

---

## Training

### Train Intent Classifier

```bash
modal run modal_app_hybrid.py
```

### Train Text To SQL Model

```bash
modal run ext_agent.py
```

### Train Qwen Extractor

```bash
modal run sql_agent.py
```

---

## Concepts Demonstrated

- Transformer Architectures
- Encoder Models
- Encoder Decoder Models
- Decoder Only Models
- LoRA Fine Tuning
- QLoRA Fine Tuning
- Quantization
- Instruction Tuning
- Production Deployment

---

## Future Enhancements

- RAG Integration
- Vector Database Support
- LangGraph Workflows
- Agentic AI
- Model Evaluation
- MLOps Pipelines

---

## Author

**Chandan Pattanayak**

Senior Principal Engineer | AI Platform Architect

- GitHub: https://github.com/CPattanayak
- LinkedIn: https://linkedin.com/in/YOUR_PROFILE