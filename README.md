# Chat1

This repository contains two projects:

1. **NLP Chatbot** — a lightweight intent-matching chatbot built with NLTK and scikit-learn.
2. **DistilBERT News Classifier** — an industry-grade deep learning text classifier fine-tuned on the AG News dataset.

---

## Project 1 — NLP Chatbot

A chatbot with Natural Language Processing (NLP) capabilities built using Python, NLTK, and scikit-learn.

## Features

- **Natural Language Understanding**: Uses NLP techniques to understand user intent
- **Intent Recognition**: Identifies user intentions using TF-IDF vectorization and cosine similarity
- **Text Preprocessing**: Lemmatization and tokenization for better understanding
- **Multiple Intents**: Supports greetings, farewells, help requests, jokes, and more
- **Interactive Chat**: Engage in real-time conversations
- **Easy to Extend**: Simple structure to add new intents and responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ankit3010droid/chat1.git
cd chat1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Chat Mode

Run the chatbot in interactive mode:
```bash
python chatbot.py
```

This will start a conversation where you can type messages and receive responses. Type `quit`, `exit`, or `bye` to end the session.

### Programmatic Usage

Use the chatbot in your own Python code:
```python
from chatbot import NLPChatbot

# Create a chatbot instance
bot = NLPChatbot()

# Get a response for a user input
response = bot.get_response("Hello!")
print(response)
```

### Demo Script

Run the demo to see example conversations:
```bash
python demo.py
```

## How It Works

1. **Text Preprocessing**: User input is converted to lowercase, special characters are removed, and words are lemmatized
2. **Intent Recognition**: The preprocessed text is converted to a TF-IDF vector and compared with trained patterns using cosine similarity
3. **Response Generation**: Once an intent is identified, a random response from that intent category is selected

## Supported Intents

- **Greeting**: hello, hi, hey, good morning, etc.
- **Goodbye**: bye, goodbye, see you, farewell, etc.
- **Thanks**: thank you, thanks, appreciate it, etc.
- **Help**: help, can you help me, what can you do, etc.
- **Name**: what is your name, who are you, etc.
- **How are you**: how are you, how's it going, etc.
- **Weather**: weather questions (explains limitation)
- **Joke**: tell me a joke, make me laugh, etc.
- **Time**: time-related queries (explains limitation)

## Extending the Chatbot

To add new intents, modify the `_load_intents()` method in `chatbot.py`:

```python
"your_intent": {
    "patterns": [
        "pattern1", "pattern2", "pattern3"
    ],
    "responses": [
        "response1",
        "response2"
    ]
}
```

---

## Project 2 — Industry-Grade News Article Classification with DistilBERT

### Overview

A portfolio-ready deep learning project that fine-tunes **DistilBERT** on the
[AG News](https://huggingface.co/datasets/ag_news) dataset to classify news articles
into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

| | |
|---|---|
| **Dataset** | AG News (HuggingFace) — 40,000 train / 7,600 test samples |
| **Model** | `distilbert-base-uncased` + linear classification head |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Training** | 3 epochs, AdamW, linear warm-up schedule, fp16 on GPU |
| **Target accuracy** | ~94 % on the AG News test set |

### Folder Structure

```text
chat1/
├── data/               # Auto-downloaded via HuggingFace Datasets
├── src/
│   ├── preprocess.py   # Tokenisation & PyTorch Dataset wrapper
│   ├── train.py        # End-to-end training pipeline
│   ├── evaluate.py     # Accuracy, F1, confusion matrix
│   ├── inference.py    # Load model & predict on new text
│   └── utils.py        # Shared helpers (seed, device, label map)
├── models/
│   └── distilbert/     # Saved model weights (after training)
├── notebooks/
│   └── colab_demo.ipynb  # End-to-end Google Colab walkthrough
├── requirements.txt
└── README.md
```

### Model Architecture

- **DistilBERT** is a distilled version of BERT: 40 % fewer parameters, 60 % faster,
  while retaining ~97 % of BERT's performance.
- A single linear classification head on top of the `[CLS]` token embedding maps
  the 768-dimensional representation to 4 class logits.
- Pre-trained weights are fine-tuned with a low learning rate (2e-5) to preserve
  the rich linguistic knowledge learned during pre-training.

### Training

```bash
# Trains on 40k AG News samples and saves the model to models/distilbert.pt
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

Sample output:
```
Accuracy : 0.9418
F1 Score : 0.9418  (weighted)

Classification Report:
              precision    recall  f1-score   support
       World       0.95      0.93      0.94      1900
      Sports       0.99      0.99      0.99      1900
    Business       0.91      0.93      0.92      1900
    Sci/Tech       0.93      0.93      0.93      1900
```

### Inference

```python
from src.inference import predict

print(predict("Apple announces new iPhone with advanced AI features."))
# → 'Sci/Tech'

print(predict("Manchester United beats Arsenal 3-1 in the Premier League."))
# → 'Sports'
```

```bash
# Or run the built-in demo
python src/inference.py
```

### Google Colab

Open `notebooks/colab_demo.ipynb` in Google Colab for a full end-to-end walkthrough
with GPU training enabled.

1. Go to *Runtime → Change runtime type → T4 GPU*
2. Run all cells in order

### References

- [AG News dataset (HuggingFace)](https://huggingface.co/datasets/ag_news)
- [DistilBERT paper](https://arxiv.org/abs/1910.01108)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

## Installation

```bash
git clone https://github.com/ankit3010droid/chat1.git
cd chat1
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- torch ≥ 2.0
- transformers ≥ 4.35
- datasets ≥ 2.14
- scikit-learn ≥ 1.3
- nltk 3.8.1
- numpy ≥ 1.24

## License

This project is open source and available for educational purposes.
