"""
A chatbot with Natural Language Processing capabilities.
Uses NLTK for text processing and sklearn for intent classification.
"""

import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class NLPChatbot:
    """A chatbot that uses NLP techniques for understanding user input."""
    
    def __init__(self):
        """Initialize the chatbot and download required NLTK data."""
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.intents = self._load_intents()
        self._download_nltk_data()
        self._prepare_training_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
    
    def _load_intents(self):
        """Load predefined intents and responses."""
        return {
            "greeting": {
                "patterns": [
                    "hello", "hi", "hey", "greetings", "good morning", 
                    "good afternoon", "good evening", "what's up", "howdy"
                ],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Greetings! How may I assist you?",
                    "Hey! What brings you here today?"
                ]
            },
            "goodbye": {
                "patterns": [
                    "bye", "goodbye", "see you", "farewell", "take care",
                    "catch you later", "until next time", "have a good day"
                ],
                "responses": [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! Come back soon!",
                    "Bye! Hope to chat with you again!"
                ]
            },
            "thanks": {
                "patterns": [
                    "thank you", "thanks", "appreciate it", "thank you very much",
                    "thanks a lot", "many thanks", "thx"
                ],
                "responses": [
                    "You're welcome!",
                    "Happy to help!",
                    "My pleasure!",
                    "Anytime! Glad I could assist!"
                ]
            },
            "help": {
                "patterns": [
                    "help", "can you help me", "i need help", "assist me",
                    "support", "what can you do", "how do you work"
                ],
                "responses": [
                    "I'm here to help! I can answer questions, have conversations, and assist you with various topics.",
                    "I can help you with many things! Just ask me a question or tell me what you need.",
                    "I'm a chatbot designed to assist you. Feel free to ask me anything!"
                ]
            },
            "name": {
                "patterns": [
                    "what is your name", "who are you", "what should i call you",
                    "what are you called", "your name"
                ],
                "responses": [
                    "I'm an NLP-powered chatbot! You can call me ChatBot.",
                    "I'm ChatBot, your AI assistant with natural language processing capabilities.",
                    "My name is ChatBot. I'm here to help you!"
                ]
            },
            "howareyou": {
                "patterns": [
                    "how are you", "how are you doing", "how's it going",
                    "how do you do", "what's up", "how are things"
                ],
                "responses": [
                    "I'm doing great, thank you for asking! How can I help you?",
                    "I'm functioning perfectly! What can I do for you today?",
                    "All systems operational! How may I assist you?"
                ]
            },
            "weather": {
                "patterns": [
                    "weather", "what's the weather", "is it raining", "temperature",
                    "forecast", "will it rain", "sunny"
                ],
                "responses": [
                    "I don't have real-time weather data, but I suggest checking a weather website or app for accurate information.",
                    "For current weather conditions, I recommend checking your local weather service.",
                    "I'm not connected to weather services, but you can easily find weather information online!"
                ]
            },
            "joke": {
                "patterns": [
                    "tell me a joke", "joke", "make me laugh", "something funny",
                    "humor", "funny"
                ],
                "responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "What do you call a bear with no teeth? A gummy bear!",
                    "Why did the scarecrow win an award? He was outstanding in his field!",
                    "What do you call fake spaghetti? An impasta!"
                ]
            },
            "time": {
                "patterns": [
                    "what time is it", "current time", "time", "what's the time",
                    "tell me the time"
                ],
                "responses": [
                    "I don't have access to real-time data, but you can check your device's clock!",
                    "I can't tell time, but your device surely can!",
                    "For the current time, please check your system clock."
                ]
            },
            "default": {
                "patterns": [],
                "responses": [
                    "I'm not sure I understand. Could you rephrase that?",
                    "Interesting! Can you tell me more?",
                    "I'm still learning. Can you ask that in a different way?",
                    "That's a good question! I'm working on understanding more queries like that."
                ]
            }
        }
    
    def _preprocess_text(self, text):
        """
        Preprocess text by converting to lowercase, removing special characters,
        and lemmatizing words.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Lemmatize
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(lemmatized)
    
    def _prepare_training_data(self):
        """Prepare training data from intents."""
        self.training_sentences = []
        self.training_labels = []
        
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                processed = self._preprocess_text(pattern)
                self.training_sentences.append(processed)
                self.training_labels.append(intent)
        
        # Fit the vectorizer
        if self.training_sentences:
            self.vectorizer.fit(self.training_sentences)
    
    def _get_intent(self, user_input):
        """
        Determine the intent of user input using TF-IDF and cosine similarity.
        """
        processed_input = self._preprocess_text(user_input)
        
        # Transform input and training sentences
        input_vector = self.vectorizer.transform([processed_input])
        training_vectors = self.vectorizer.transform(self.training_sentences)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, training_vectors)[0]
        
        # Get the best match
        max_similarity = np.max(similarities)
        
        # Threshold for intent recognition
        if max_similarity > 0.3:
            best_match_idx = np.argmax(similarities)
            return self.training_labels[best_match_idx]
        
        return "default"
    
    def get_response(self, user_input):
        """
        Get a response for the user input.
        """
        if not user_input or not user_input.strip():
            return "Please say something!"
        
        intent = self._get_intent(user_input)
        responses = self.intents[intent]["responses"]
        
        # Return a random response from the intent
        return np.random.choice(responses)
    
    def chat(self):
        """
        Start an interactive chat session.
        """
        print("="*50)
        print("NLP Chatbot - Type 'quit', 'exit', or 'bye' to end")
        print("="*50)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("ChatBot: Goodbye! Have a great day!")
                    break
                
                response = self.get_response(user_input)
                print(f"ChatBot: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\nChatBot: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


def main():
    """Main function to run the chatbot."""
    print("Initializing NLP Chatbot...")
    chatbot = NLPChatbot()
    print("Chatbot ready!")
    print()
    chatbot.chat()


if __name__ == "__main__":
    main()
