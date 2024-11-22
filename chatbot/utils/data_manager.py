import pandas as pd
from transformers import AutoTokenizer

def load_and_preprocess_data(file_path, tokenizer, max_length=512):
    """
    Charge les données depuis un fichier CSV et prépare les paires question-réponse
    pour le fine-tuning du modèle DialoGPT.

    Args:
        file_path (str): Le chemin du fichier CSV contenant les données (avec des colonnes 'Question' et 'Answer').
        tokenizer (transformers.tokenizer): Le tokenizer utilisé pour la tokenisation.
        max_length (int): La longueur maximale des séquences.

    Returns:
        dict: Un dictionnaire contenant les entrées tokenisées prêtes pour l'entraînement.
    """
    # Charger les données
    data = pd.read_csv(file_path)
    
    # Liste pour stocker les paires question-réponse
    conversation_pairs = []
    
    # Ajouter le token EOS entre la question et la réponse
    for _, row in data.iterrows():
        question = row['Question']
        answer = row['Answer']
        conversation_pairs.append(f"{question} {tokenizer.eos_token} {answer} {tokenizer.eos_token}")
    
    # Tokenisation des paires question-réponse
    encodings = tokenizer(conversation_pairs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    return encodings
