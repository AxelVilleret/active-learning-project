import numpy as np
import pandas as pd
import os
import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from global_variables import *
from utils import clean_text, tokenize_words
import matplotlib.pyplot as plt

def create_histogram(labels, values, bins=None):
    # Vérifier si les labels sont uniques
    if len(labels) != len(set(labels)):
        print("Erreur : Les labels doivent être uniques.")
        return

    # Vérifier si les deux tableaux ont la même longueur
    if len(labels) != len(values):
        print("Erreur : Les deux tableaux doivent avoir la même longueur.")
        return

    # Créer un histogramme
    plt.bar(labels, values)

    # Ajouter un titre
    plt.title('Classes Distribution')

    # Afficher l'histogramme
    plt.show()


def create_dict():

    vocab = []
    df_food = pd.read_csv(FOOD_DATASET_PATH)

    df_cloth = pd.read_csv(CLOTH_DATASET_PATH)

    for i in tqdm.tqdm(range(len(df_food)), "Cleaning X"):
        target = clean_text(df_food['Text'].loc[i]).split()
        for word in target:
            vocab.append(word)

    for i in tqdm.tqdm(range(len(df_cloth)), "Cleaning X"):
        target = clean_text(df_cloth['Review Text'].loc[i]).split()
        for word in target:
            vocab.append(word)

     # vocab = set(vocab)
    vocab = Counter(vocab)

    # delete words that occur less than 10 times
    vocab = {k: v for k, v in vocab.items() if v >= 10}

    # word to integer encoder dict
    vocab2int = {word: i for i, word in enumerate(vocab, start=1)}
    # pickle int2vocab for testing
    print("Pickling vocab2int...")
    pickle.dump(vocab2int, open(VOCAB2INT, "wb"))


def get_dict():
    if not os.path.exists(VOCAB2INT):
        create_dict()

    vocab2int = pickle.load(open(VOCAB2INT, "rb"))
    return vocab2int


def load_food_review_data():

    df = pd.read_csv(FOOD_DATASET_PATH)
    X = np.zeros((len(df)//FRACTION_OF_FOOD, 2), dtype=object)
    print(len(X))
    for i in tqdm.tqdm(range(len(df)//FRACTION_OF_FOOD), "Cleaning X"):
        target = df['Text'].loc[i]
        X[i, 0] = clean_text(target)
        X[i, 1] = df['Score'].loc[i]-1

    # create_histogram(np.unique(X[:, 1]), np.bincount(X[:, 1].astype(int)).astype(np.int32))

    vocab2int = get_dict()

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(X[i, 0], vocab2int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X[:, 0], X[:, 1], test_size=0.2, random_state=19)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=19)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, vocab2int


def load_cloth_review_data():
    df = pd.read_csv(CLOTH_DATASET_PATH)
    print(len(df))
    X = np.zeros((len(df), 2), dtype=object)
    for i in tqdm.tqdm(range(len(df)), "Cleaning X"):
        target = df['Review Text'].loc[i]
        rating = df['Rating'].loc[i]
        if not target == "empty":
            X[i, 0] = clean_text(target)
            X[i, 1] = rating-1

    lignes_non_zero = np.any(X != 0, axis=1)
    X = X[lignes_non_zero]
    # create_histogram(np.unique(X[:, 1]), np.bincount(X[:, 1].astype(int)).astype(np.int32))

    vocab2int = get_dict()

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(X[i, 0], vocab2int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X[:, 0], X[:, 1], test_size=0.2, random_state=19)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=19)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, vocab2int

if __name__ == "__main__":
    # load_food_review_data()
    load_cloth_review_data()