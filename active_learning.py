import glob
from preprocess import load_cloth_review_data, load_food_review_data
import numpy as np
from keras.models import load_model, clone_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from config import sequence_length, embedding_size, batch_size, epochs
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from datetime import datetime
from sklearn.cluster import KMeans
from visualization import update_json
from collections import Counter
import pandas as pd
from global_variables import *

X_train, X_validation, X_test, y_train, y_validation, y_test, vocab = load_cloth_review_data()

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
y_train = y_train.astype(np.int32)
X_validation = sequence.pad_sequences(X_validation, maxlen=sequence_length)
y_validation = y_validation.astype(np.int32)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)
y_test = y_test.astype(np.int32)

pretrained_model = load_model(PRETRAINED_MODEL_PATH)

def convert_pourcentage_to_quantity(pourcentage):
    return int(pourcentage * len(X_train) / 100)

def select_samples_randomly(quantity):
    # Select samples randomly
    random_indices = np.random.choice(len(X_train), quantity, replace=False)
    X_train_random = X_train[random_indices]
    y_train_random = y_train[random_indices]
    return X_train_random, y_train_random

def select_samples_by_clustering(quantity):
    NB_CLUSTERS = quantity

    # Utilisez KMeans pour regrouper les données non étiquetées
    kmeans = KMeans(n_clusters=NB_CLUSTERS, random_state=0).fit(X_train)
    quantity_per_cluster = int(quantity / NB_CLUSTERS)

    X_train_clustering = []
    y_train_clustering = []

    # Pour chaque cluster, sélectionnez l'échantillon le plus proche du centroïde
    for j in range(NB_CLUSTERS):
        cluster_indices = np.where(kmeans.labels_ == j)[0]
        centroid = kmeans.cluster_centers_[j]
        distances = np.linalg.norm(X_train[cluster_indices] - centroid, axis=1)
        closest_indices = np.argsort(distances)[:quantity_per_cluster]
        X_train_clustering.extend(X_train[cluster_indices[closest_indices]])
        y_train_clustering.extend(y_train[cluster_indices[closest_indices]])

    return np.array(X_train_clustering), np.array(y_train_clustering)

def select_samples_by_representative_sampling(quantity):
    X_train_food, X_validation_food, X_test_food, y_train_food, y_validation_food, y_test_food, vocab_food = load_food_review_data()
    X_train_food = sequence.pad_sequences(X_train_food, maxlen=sequence_length)
    kmeans_food = KMeans(n_clusters=1, random_state=0).fit(X_train_food)
    centroid_food = kmeans_food.cluster_centers_[0]
    kmeans_cloth = KMeans(n_clusters=1, random_state=0).fit(X_train)
    centroid_cloth = kmeans_cloth.cluster_centers_[0]
    # Sample items that have the greatest outlier score from the training relative to their outlier score from the unlabeled data (ɗ/ɗ’)
    outlier_scores = np.linalg.norm(X_train - centroid_food, axis=1) / np.linalg.norm(X_train - centroid_cloth, axis=1)
    representative_samples = np.argsort(outlier_scores)[-quantity:]
    X_train_representative_sampling = X_train[representative_samples]
    y_train_representative_sampling = y_train[representative_samples]
    return X_train_representative_sampling, y_train_representative_sampling

def select_samples_by_least_confidence(quantity):
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by least confidence
    least_confidence = np.argsort(np.max(y_pred, axis=1))
    X_train_least_confidence = X_train[least_confidence[:quantity]]
    y_train_least_confidence = y_train[least_confidence[:quantity]]
    return X_train_least_confidence, y_train_least_confidence

def select_samples_by_margin(quantity):
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by margin
    margin = np.argsort(np.max(y_pred, axis=1) - np.partition(y_pred, -2, axis=1)[:, -2])
    X_train_margin = X_train[margin[:quantity]]
    y_train_margin = y_train[margin[:quantity]]
    return X_train_margin, y_train_margin

def select_samples_by_entropy(quantity):
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by entropy
    entropy = np.argsort(-np.sum(y_pred * np.log(y_pred), axis=1))
    X_train_entropy = X_train[entropy[:quantity]]
    y_train_entropy = y_train[entropy[:quantity]]
    return X_train_entropy, y_train_entropy

def select_by_mixed_with_integrated_scores(first_method, second_method, quantity):
    quantity_with_first_method = quantity // 2
    quantity_with_second_method = quantity - quantity_with_first_method
    X_train_first_method, y_train_first_method = algorithms[first_method](quantity_with_first_method)
    X_train_second_method, y_train_second_method = algorithms[second_method](quantity_with_second_method)
    X_train_mixed, y_train_mixed = np.concatenate((X_train_first_method, X_train_second_method)), np.concatenate((y_train_first_method, y_train_second_method))
    return X_train_mixed, y_train_mixed

def select_by_mixed_with_least_confidence_and_representative_sampling(quantity):
    return select_by_mixed_with_integrated_scores(LEAST_CONFIDENCE, REPRESENTATIVE_SAMPLING, quantity)

def select_by_mixed_with_margin_and_clustering(quantity):
    return select_by_mixed_with_integrated_scores(MARGIN, CLUSTERING, quantity)

def retrain_model(X_train_selected, y_train_selected, algorithm, pourcentage):
    print(f"Retraining model with this number of samples: {len(X_train_selected)}")
    model = load_model(PRETRAINED_MODEL_PATH)
    checkpointer = ModelCheckpoint(f"models/{algorithm}_{pourcentage}.h5", save_best_only=True, verbose=1)
    history = model.fit(X_train_selected, y_train_selected, epochs=epochs, validation_data=(
        X_validation, y_validation), batch_size=batch_size, callbacks=[checkpointer])
    # model.save(f"models/{algorithm}_{pourcentage}.h5")
    return history

def evaluate_model(algorithm, pourcentage):
    model_path = f'models/{algorithm}_{pourcentage}.h5'
    model = load_model(model_path)
    
    # Evaluate the model on the test data
    loss, acc = model.evaluate(X_test, y_test)
    result = f'{algorithm} - {pourcentage} - {acc:.4f}'
    print(result)
    return loss, acc


algorithms = {
    RANDOM: select_samples_randomly,
    CLUSTERING: select_samples_by_clustering,
    REPRESENTATIVE_SAMPLING: select_samples_by_representative_sampling,
    LEAST_CONFIDENCE: select_samples_by_least_confidence,
    MARGIN: select_samples_by_margin,
    # ENTROPY: select_samples_by_entropy,
    # MIXED_WITH_LEAST_CONFIDENCE_AND_REPRESENTATIVE_SAMPLING: select_by_mixed_with_least_confidence_and_representative_sampling,
    # MIXED_WITH_MARGIN_AND_CLUSTERING: select_by_mixed_with_margin_and_clustering,
}

pourcentages = [ 
    1,
    5,
    10, 
    15, 
    20, 
]

# pourcentages used to determine the number of samples to select
# pourcentages = [
#     1,
#     2,
#     4,
#     8,
#     16,
#     32,
# ]

selected_samples = {}


def pourcentage_lignes_communes(matrice1, matrice2):
    # Convertir les lignes en tuples pour qu'elles soient hashables
    lignes_matrice1 = set(tuple(ligne) for ligne in matrice1)
    lignes_matrice2 = set(tuple(ligne) for ligne in matrice2)
    # Compter le nombre de lignes communes
    lignes_communes = len(lignes_matrice1.intersection(lignes_matrice2))
    # Calculer le pourcentage
    pourcentage = lignes_communes / len(matrice1) * 100
    return pourcentage

def in_common_pourcentages():
    global selected_samples
    algorithms = list(selected_samples.keys()) 
    for i in range(len(pourcentages)):
        current_matrix = pd.DataFrame(columns=algorithms, index=algorithms)
        for algo_1 in algorithms:
            for algo_2 in algorithms:
                current_matrix[algo_1][algo_2] = pourcentage_lignes_communes(selected_samples[algo_1][pourcentages[i]], selected_samples[algo_2][pourcentages[i]])
        current_matrix.to_csv(f"results/in_common_{pourcentages[i]}.csv")


def main():
    base_loss, base_acc = evaluate_model(BASE, 0)
    update_json(RESULTS_PATH, BASE, 0, base_acc, base_loss, None)
    for algorithm in algorithms:
        selected_samples[algorithm] = {}
        for pourcentage in pourcentages:
            quantity = convert_pourcentage_to_quantity(pourcentage)
            X_train_selected, y_train_selected = algorithms[algorithm](quantity)
            selected_samples[algorithm][pourcentage] = X_train_selected
            history = retrain_model(X_train_selected, y_train_selected, algorithm, pourcentage)
            loss, acc = evaluate_model(algorithm, pourcentage)
            update_json(RESULTS_PATH, algorithm, pourcentage, acc, loss, history.history)
    # in_common_pourcentages()

if __name__ == "__main__":
    main()
