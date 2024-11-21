from chatbot.model.fine_tuning import FineTuner
from chatbot.utils.config import Config

if __name__ == "__main__":
    fine_tuner = FineTuner(model_name=Config.MODEL_NAME, output_dir="./fine_tuned_model")
    fine_tuner.fine_tune(dataset_path="data/processed/cleaned_data.json", epochs=3, batch_size=8)

