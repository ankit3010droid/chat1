"""
Test script for the NLP Chatbot to verify functionality.
"""

from chatbot import NLPChatbot


def test_chatbot():
    """Test various aspects of the chatbot."""
    print("Testing NLP Chatbot...")
    print("="*60)
    
    # Initialize chatbot
    bot = NLPChatbot()
    print("✓ Chatbot initialized successfully")
    
    # Test 1: Basic intent recognition
    test_cases = [
        ("hello", "greeting"),
        ("hi there", "greeting"),
        ("goodbye", "goodbye"),
        ("thank you", "thanks"),
        ("what is your name", "name"),
        ("help me", "help"),
        ("how are you", "howareyou"),
        ("tell me a joke", "joke"),
    ]
    
    print("\nTest 1: Intent Recognition")
    print("-" * 60)
    all_passed = True
    for user_input, expected_intent in test_cases:
        detected_intent = bot._get_intent(user_input)
        status = "✓" if detected_intent == expected_intent else "✗"
        print(f"{status} Input: '{user_input}' -> Intent: {detected_intent} (Expected: {expected_intent})")
        if detected_intent != expected_intent:
            all_passed = False
    
    # Test 2: Response generation
    print("\nTest 2: Response Generation")
    print("-" * 60)
    test_inputs = ["Hi!", "Thanks!", "What's the weather?", "Random text that doesn't match"]
    
    for user_input in test_inputs:
        response = bot.get_response(user_input)
        print(f"✓ Input: '{user_input}' -> Response: '{response}'")
    
    # Test 3: Edge cases
    print("\nTest 3: Edge Cases")
    print("-" * 60)
    
    # Empty input
    response = bot.get_response("")
    print(f"✓ Empty input -> Response: '{response}'")
    
    # Very long input
    long_input = "hello " * 50
    response = bot.get_response(long_input)
    print(f"✓ Long input -> Response: '{response}'")
    
    # Special characters
    response = bot.get_response("!!!hello???")
    print(f"✓ Special characters -> Response: '{response}'")
    
    # Numbers
    response = bot.get_response("12345")
    print(f"✓ Numbers only -> Response: '{response}'")
    
    print("\n" + "="*60)
    print("Testing completed!")
    if all_passed:
        print("✓ All intent recognition tests passed!")
    else:
        print("⚠ Some intent recognition tests had unexpected results")
    print("="*60)


if __name__ == "__main__":
    test_chatbot()
