from chatbot.model.dialo_gpt import DialoGPTChatbot


class Chatbot:
    """Handles user interaction with the chatbot."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initializes the chatbot with DialoGPT."""
        self.chatbot = DialoGPTChatbot(model_name)

    def start_conversation(self):
        """Starts a conversation with the user."""
        print("Chatbot: Hello! How can I assist you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break
            response = self.chatbot.generate_response(user_input)
            print(f"Chatbot: {response}")

