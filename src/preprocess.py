"""
Data preprocessing pipeline for DistilBERT text classification.

Handles:
- Dataset loading and subsampling (AG News via HuggingFace Datasets)
- DistilBERT tokenization with padding/truncation
- PyTorch Dataset wrapper for use with DataLoader
"""

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

# Maximum token length for DistilBERT input
MAX_LEN = 256

# Pretrained tokenizer name
TOKENIZER_NAME = "distilbert-base-uncased"


class NewsDataset(Dataset):
    """
    PyTorch Dataset that wraps a list of {'text': str, 'label': int} samples
    and returns DistilBERT-compatible encodings.
    """

    def __init__(self, samples: list, tokenizer: DistilBertTokenizerFast, max_len: int = MAX_LEN):
        """
        Args:
            samples:   List of dicts with keys 'text' and 'label'.
            tokenizer: Pre-loaded DistilBertTokenizerFast instance.
            max_len:   Maximum token sequence length (padding/truncation applied).
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        # Tokenize with padding and truncation to a fixed length
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Squeeze the batch dimension added by return_tensors='pt'
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def load_tokenizer(tokenizer_name: str = TOKENIZER_NAME) -> DistilBertTokenizerFast:
    """Load and return the DistilBERT fast tokenizer."""
    return DistilBertTokenizerFast.from_pretrained(tokenizer_name)


def build_dataset(hf_dataset, tokenizer: DistilBertTokenizerFast, max_len: int = MAX_LEN) -> NewsDataset:
    """
    Convert a HuggingFace Dataset split into a NewsDataset.

    Args:
        hf_dataset: A HuggingFace Dataset object with 'text' and 'label' columns.
        tokenizer:  Pre-loaded DistilBertTokenizerFast.
        max_len:    Maximum token sequence length.

    Returns:
        NewsDataset instance ready for DataLoader.
    """
    samples = [{"text": row["text"], "label": row["label"]} for row in hf_dataset]
    return NewsDataset(samples, tokenizer, max_len)


def load_ag_news(train_size: int = 40_000, seed: int = 42):
    """
    Download and subsample the AG News dataset from HuggingFace.

    Args:
        train_size: Number of training samples to keep (default 40,000).
        seed:       Random seed for shuffling.

    Returns:
        Tuple of (train_split, test_split) as HuggingFace Dataset objects.
    """
    from datasets import load_dataset  # imported here to keep module importable without 'datasets' installed

    dataset = load_dataset("ag_news")

    # Subsample training split for Colab free-tier compatibility
    train_split = dataset["train"].shuffle(seed=seed).select(range(train_size))
    test_split = dataset["test"]  # ~7,600 samples — used as-is

    return train_split, test_split
