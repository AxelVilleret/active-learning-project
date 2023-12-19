import pandas as pd

# Votre liste
ma_liste = ['representative_sampling', 'margin_of_confidence']

# Création d'un DataFrame vide avec les colonnes et les index nommés d'après votre liste
df = pd.DataFrame(columns=ma_liste, index=ma_liste)

# df.loc['representative_sampling', 'margin_of_confidence'] = votre_valeur
# Remplir toutes les valeurs avec des 1 en utilisant fillna()
df = df.fillna(1)



print(df)
# Sauvegarde du DataFrame dans un fichier CSV
df.to_csv('mon_dataframe.csv')
