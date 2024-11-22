from chatbot.core.conversation import Chatbot
from chatbot.utils.config import Config
from chatbot.core.logger import setup_logger
from chatbot.model.fine_tuning import fine_tune_model
import os

if __name__ == "__main__":
    logger = setup_logger('chatbot', Config.LOG_FILE)
    
    # Fine-tuning du modèle si nécessaire
    fine_tuned_model_dir = "fine_tuned_model"
    if not os.path.exists(fine_tuned_model_dir):
        logger.info("Modèle non trouvé, démarrage du fine-tuning.")
        fine_tune_model('data/raw/AI.csv', output_dir=fine_tuned_model_dir)
    else:
        logger.info("Modèle fine-tuné trouvé, pas besoin de fine-tuning.")
    
    # Initialiser et démarrer la conversation avec le chatbot
    chatbot = Chatbot(Config.MODEL_NAME, fine_tuned_model_dir=fine_tuned_model_dir)
    chatbot.start_conversation()
