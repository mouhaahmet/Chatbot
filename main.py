from chatbot.core.conversation import Chatbot
from chatbot.utils.config import Config
from chatbot.core.logger import setup_logger

if __name__ == "__main__":
    logger = setup_logger('chatbot', Config.LOG_FILE)
    chatbot = Chatbot(Config.MODEL_NAME)
    chatbot.start_conversation()

