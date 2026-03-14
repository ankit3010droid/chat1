"""
Training pipeline for DistilBERT sequence classification on AG News.

Usage (from repo root):
    python src/train.py

The trained model weights are saved to models/distilbert.pt.
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.preprocess import load_ag_news, build_dataset, load_tokenizer
from src.utils import set_seed, get_device, MODEL_PATH, NUM_LABELS

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE = 16          # Fits comfortably on Colab T4 GPU with fp16
LEARNING_RATE = 2e-5     # Standard fine-tuning LR for BERT-family models
NUM_EPOCHS = 3           # 3 epochs is sufficient for strong results on AG News
WEIGHT_DECAY = 0.01      # L2 regularization
WARMUP_RATIO = 0.1       # 10 % of total steps used for LR warm-up
MAX_LEN = 256
TRAIN_SIZE = 40_000
SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def build_model(num_labels: int = NUM_LABELS) -> DistilBertForSequenceClassification:
    """Load DistilBERT with a classification head for `num_labels` classes."""
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )


def train_epoch(model, loader, optimizer, scheduler, device, use_fp16: bool = False):
    """Run one full training epoch and return the average loss."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device):
    """Run evaluation on `loader` and return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(train_size: int = TRAIN_SIZE, seed: int = SEED):
    """
    End-to-end training function:
      1. Downloads and preprocesses AG News.
      2. Trains DistilBERT for NUM_EPOCHS.
      3. Saves the final model to MODEL_PATH.

    Returns:
        Trained DistilBertForSequenceClassification model.
    """
    set_seed(seed)
    device = get_device()
    use_fp16 = device.type == "cuda"
    print(f"Device: {device} | Mixed precision: {use_fp16}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading AG News dataset …")
    train_hf, test_hf = load_ag_news(train_size=train_size, seed=seed)

    tokenizer = load_tokenizer()
    train_dataset = build_dataset(train_hf, tokenizer, max_len=MAX_LEN)
    eval_dataset = build_dataset(test_hf, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model & Optimiser ─────────────────────────────────────────────────────
    model = build_model().to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training — {NUM_EPOCHS} epochs, {len(train_loader)} steps/epoch")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_fp16)
        eval_loss, eval_acc = evaluate_epoch(model, eval_loader, device)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {eval_loss:.4f}  |  "
            f"Val Acc: {eval_acc:.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train()
