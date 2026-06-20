# Hybrid AI Training Platform

This repository demonstrates practical fine-tuning patterns for three major transformer architectures:

| Architecture | Model | Use Case |
|--------------|--------|----------|
| Encoder | DistilBERT | Intent Classification |
| Encoder-Decoder | FLAN-T5 | Text-to-SQL |
| Decoder-Only | Qwen 2.5 | Information Extraction |

## Why This Repository?

Many engineers learn how to call LLM APIs but never understand:

- How encoder models are trained
- How sequence-to-sequence models work
- How decoder-only LLMs are fine-tuned
- When to use LoRA vs QLoRA

This repository provides working examples for all three.

## Architecture

![Architecture](docs/architecture.png)

## Encoder Training (DistilBERT)

![Encoder](docs/encoder-training.png)

### Use Cases

- Intent Classification
- Text Classification
- Document Categorization

## Encoder-Decoder Training (FLAN-T5)

![Encoder Decoder](docs/encoder-decoder-training.png)

### Use Cases

- Text-to-SQL
- Translation
- Summarization

## Decoder Only Training (Qwen)

![Decoder Only](docs/decoder-only-training.png)

### Use Cases

- Information Extraction
- Chat Applications
- JSON Generation

## Technologies

- Hugging Face
- PEFT
- LoRA
- QLoRA
- TRL
- Modal
- FastAPI

## Training

```bash
modal run modal_app_hybrid.py
modal run ext_agent.py
modal run sql_agent.py
```

## Author

Chandan Pattanayak

Senior Principal Engineer 