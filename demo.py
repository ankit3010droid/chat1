"""
Demo script showing how to use the NLP Chatbot programmatically.
"""

from chatbot import NLPChatbot


def demo():
    """Demonstrate the chatbot's capabilities."""
    print("="*60)
    print("NLP Chatbot Demo")
    print("="*60)
    print("\nInitializing chatbot...")
    
    # Create chatbot instance
    bot = NLPChatbot()
    print("Chatbot initialized successfully!")
    
    # Example conversations
    test_inputs = [
        "Hello!",
        "How are you?",
        "What's your name?",
        "Can you help me?",
        "Tell me a joke",
        "What's the weather like?",
        "Thank you!",
        "Goodbye"
    ]
    
    print("\n" + "="*60)
    print("Demo Conversation:")
    print("="*60 + "\n")
    
    for user_input in test_inputs:
        print(f"You: {user_input}")
        response = bot.get_response(user_input)
        print(f"ChatBot: {response}")
        print()
    
    print("="*60)
    print("Demo completed!")
    print("="*60)
    print("\nTo start an interactive chat session, run: python chatbot.py")


if __name__ == "__main__":
    demo()
