import csv

# Ouvrir le fichier csv
with open('data/ReviewsCloth.csv', 'r') as fichier:
    lecteur = csv.reader(fichier)
    lignes = list(lecteur)

# Calculer l'index de séparation
index_separation = int(len(lignes) * 0.8)

# Créer le fichier active_learning_set.csv
with open('data/active_learning_set.csv', 'w', newline='') as fichier:
    ecrivain = csv.writer(fichier)
    # +1 pour inclure la ligne d'index_separation
    for ligne in lignes[:index_separation+1]:
        ecrivain.writerow(ligne)

# Créer le fichier test_set.csv
with open('data/test_set.csv', 'w', newline='') as fichier:
    ecrivain = csv.writer(fichier)
    ecrivain.writerow(lignes[0])  # écrire l'en-tête
    for ligne in lignes[index_separation+1:]:
        ecrivain.writerow(ligne)
