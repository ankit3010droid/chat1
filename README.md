# Chat1 - NLP Chatbot

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

## Requirements

- Python 3.7+
- nltk 3.8.1
- numpy >= 1.24.0
- scikit-learn >= 1.3.0

## License

This project is open source and available for educational purposes.
