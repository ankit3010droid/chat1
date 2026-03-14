"""
Evaluation module for the DistilBERT text classifier.

Computes accuracy, weighted F1 score, and a confusion matrix against the
AG News test set, then prints a formatted classification report.

Usage (from repo root):
    python src/evaluate.py
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.preprocess import load_ag_news, build_dataset, load_tokenizer
from src.utils import get_device, MODEL_PATH, LABEL_NAMES

BATCH_SIZE = 32


def get_predictions(model: DistilBertForSequenceClassification, loader: DataLoader, device: torch.device):
    """
    Run inference over all batches and collect predictions and ground-truth labels.

    Returns:
        Tuple of (predictions list, true_labels list).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_preds, all_labels


def evaluate(model_path: str = MODEL_PATH) -> dict:
    """
    Load the saved model, run evaluation on the AG News test set, and print metrics.

    Args:
        model_path: Directory containing the saved model (output of model.save_pretrained).

    Returns:
        Dict with keys 'accuracy', 'f1', 'confusion_matrix'.
    """
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = load_tokenizer()

    # ── Load test data ────────────────────────────────────────────────────────
    print("Loading test data …")
    _, test_hf = load_ag_news()
    test_dataset = build_dataset(test_hf, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Get predictions ───────────────────────────────────────────────────────
    print("Running inference …")
    predictions, true_labels = get_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    cm = confusion_matrix(true_labels, predictions)

    label_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

    print("\n" + "=" * 55)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}  (weighted)")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_names))
    print("Confusion Matrix:")
    print(cm)

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}


if __name__ == "__main__":
    evaluate()
