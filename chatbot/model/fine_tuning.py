import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AdamW
from torch.utils.data import Dataset, DataLoader
from chatbot.utils.data_manager import load_and_preprocess_data
import os

# Charger le tokenizer et le modèle pré-entraîné
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Préparer le Dataset
class ChatbotDataset(Dataset):
    def __init__(self, encodings, tokenizer):
        self.encodings = encodings
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# Fine-tuning du modèle
def fine_tune_model(file_path, output_dir='fine_tuned_model', epochs=3, batch_size=4, learning_rate=5e-5):
    # Charger et préparer les données avec la fonction du module utils
    encodings = load_and_preprocess_data(file_path, tokenizer)
    dataset = ChatbotDataset(encodings, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimiseur
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Appareil (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fine-tuning
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            # Calculer la perte
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader)}")

    # Sauvegarder le modèle fine-tuné
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modèle fine-tuné sauvegardé dans {output_dir}")

# Exemple d'appel de la fonction de fine-tuning
if __name__ == "__main__":
    fine_tune_model('data/raw/AI.csv')
