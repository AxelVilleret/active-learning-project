from preprocess import load_cloth_review_data, load_food_review_data
import numpy as np
from keras.models import load_model, clone_model
from keras.preprocessing import sequence
from sklearn.cluster import KMeans
from global_variables import *


def select_samples_randomly(quantity, X_train, y_train):
    # Select samples randomly
    random_indices = np.random.choice(len(X_train), quantity, replace=False)
    X_train_random = X_train[random_indices]
    y_train_random = y_train[random_indices]
    return X_train_random, y_train_random


def select_samples_by_clustering(quantity, X_train, y_train):
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


def select_samples_by_representative_sampling(quantity, X_train, y_train):
    X_train_food, X_validation_food, X_test_food, y_train_food, y_validation_food, y_test_food, vocab_food = load_food_review_data()
    X_train_food = sequence.pad_sequences(X_train_food, maxlen=SEQUENCE_LENGTH)
    kmeans_food = KMeans(n_clusters=1, random_state=0).fit(X_train_food)
    centroid_food = kmeans_food.cluster_centers_[0]
    kmeans_cloth = KMeans(n_clusters=1, random_state=0).fit(X_train)
    centroid_cloth = kmeans_cloth.cluster_centers_[0]
    # Sample items that have the greatest outlier score from the training relative to their outlier score from the unlabeled data (ɗ/ɗ’)
    outlier_scores = np.linalg.norm(
        X_train - centroid_food, axis=1) / np.linalg.norm(X_train - centroid_cloth, axis=1)
    representative_samples = np.argsort(outlier_scores)[-quantity:]
    X_train_representative_sampling = X_train[representative_samples]
    y_train_representative_sampling = y_train[representative_samples]
    return X_train_representative_sampling, y_train_representative_sampling


def select_samples_by_least_confidence(quantity, X_train, y_train):
    pretrained_model = load_model(PRETRAINED_MODEL_PATH)
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by least confidence
    least_confidence = np.argsort(np.max(y_pred, axis=1))
    X_train_least_confidence = X_train[least_confidence[:quantity]]
    y_train_least_confidence = y_train[least_confidence[:quantity]]
    return X_train_least_confidence, y_train_least_confidence


def select_samples_by_margin(quantity, X_train, y_train):
    pretrained_model = load_model(PRETRAINED_MODEL_PATH)
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by margin
    margin = np.argsort(np.max(y_pred, axis=1) -
                        np.partition(y_pred, -2, axis=1)[:, -2])
    X_train_margin = X_train[margin[:quantity]]
    y_train_margin = y_train[margin[:quantity]]
    return X_train_margin, y_train_margin


def select_samples_by_entropy(quantity, X_train, y_train):
    pretrained_model = load_model(PRETRAINED_MODEL_PATH)
    # Evaluate the model on the test data
    y_pred = pretrained_model.predict(X_train)
    # Select samples by entropy
    entropy = np.argsort(-np.sum(y_pred * np.log(y_pred), axis=1))
    X_train_entropy = X_train[entropy[:quantity]]
    y_train_entropy = y_train[entropy[:quantity]]
    return X_train_entropy, y_train_entropy


def select_by_mixed_with_integrated_scores(first_method, second_method, quantity, X_train, y_train):
    quantity_with_first_method = quantity // 2
    quantity_with_second_method = quantity - quantity_with_first_method
    X_train_first_method, y_train_first_method = first_method(
        quantity_with_first_method, X_train, y_train)
    X_train_second_method, y_train_second_method = second_method(
        quantity_with_second_method, X_train, y_train)
    X_train_mixed, y_train_mixed = np.concatenate((X_train_first_method, X_train_second_method)), np.concatenate(
        (y_train_first_method, y_train_second_method))
    return X_train_mixed, y_train_mixed


def select_by_mixed_with_least_confidence_and_clustering(quantity, X_train, y_train):
    return select_by_mixed_with_integrated_scores(select_samples_by_least_confidence, select_samples_by_clustering, quantity, X_train, y_train)


def select_by_mixed_with_margin_and_clustering(quantity, X_train, y_train):
    return select_by_mixed_with_integrated_scores(select_samples_by_margin, select_samples_by_clustering, quantity, X_train, y_train)
