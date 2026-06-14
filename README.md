modal volume create distilbert-intent-cache
modal volume create flan-extractor-cache
modal volume put gliner-extractor-cache extractor_dataset.jsonl
modal volume put flan-extractor-cache dataset.jsonl
modal run modal_app_hybrid.py::train_classifier
modal run modal_app_hybrid.py::train_extractor
modal deploy modal_app_hybrid.py