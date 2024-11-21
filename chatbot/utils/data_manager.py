import pandas as pd

# Charger le fichier CSV
data_path = 'data/raw/AI.csv'
data = pd.read_csv(data_path)

# Vérifier un aperçu des données
print(data.head())
