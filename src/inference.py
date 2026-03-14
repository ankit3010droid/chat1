"""
Inference module for the DistilBERT text classifier.

Loads the saved model from disk and predicts the AG News category
for arbitrary input text.

Usage (from repo root):
    python src/inference.py

Or import and call predict() programmatically:
    from src.inference import predict
    print(predict("Apple unveils the new M4 MacBook Pro."))
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.utils import get_device, MODEL_PATH, LABEL_NAMES

# Module-level cache so the model is loaded only once per process
_model = None
_tokenizer = None


def _load_model_and_tokenizer(model_path: str = MODEL_PATH):
    """Load model and tokenizer into module-level cache (lazy, one-time initialisation)."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        device = get_device()
        _tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        _model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
        _model.eval()
    return _model, _tokenizer


def predict(text: str, model_path: str = MODEL_PATH, max_len: int = 256) -> str:
    """
    Classify a single news article/snippet into one of four AG News categories.

    Args:
        text:       Raw input text (any length; will be truncated to max_len tokens).
        model_path: Path to the directory produced by model.save_pretrained().
        max_len:    Maximum token length for encoding.

    Returns:
        Human-readable category label: 'World', 'Sports', 'Business', or 'Sci/Tech'.

    Example:
        >>> from src.inference import predict
        >>> predict("SpaceX launches another Falcon 9 rocket to the ISS.")
        'Sci/Tech'
    """
    model, tokenizer = _load_model_and_tokenizer(model_path)
    device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return LABEL_NAMES[pred_id]


if __name__ == "__main__":
    # Quick smoke test with a handful of diverse samples
    samples = [
        ("Apple announces new iPhone with advanced AI features at WWDC.", "Sci/Tech"),
        ("Manchester United beat Arsenal 3-1 in the Premier League clash.", "Sports"),
        ("Federal Reserve raises interest rates amid inflation concerns.", "Business"),
        ("UN Security Council convenes emergency session over Middle East tensions.", "World"),
    ]

    print("=" * 60)
    print("DistilBERT AG News Inference Demo")
    print("=" * 60)
    for text, expected in samples:
        result = predict(text)
        status = "✓" if result == expected else "✗"
        print(f"{status}  Predicted: {result:10s}  |  Expected: {expected}")
        print(f"   Text: {text[:75]}…\n")
