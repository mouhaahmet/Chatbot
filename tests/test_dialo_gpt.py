import unittest
from chatbot.model.dialo_gpt import DialoGPTChatbot

class TestDialoGPTChatbot(unittest.TestCase):
    def test_generate_response(self):
        chatbot = DialoGPTChatbot("microsoft/DialoGPT-small")
        response = chatbot.generate_response("Hello!")
        self.assertIsInstance(response, str)

