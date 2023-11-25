from preprocess import load_cloth_review_data
import numpy as np
from keras.models import load_model, clone_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from config import sequence_length, embedding_size, batch_size, epochs
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from datetime import datetime

X_train, X_test, y_train, y_test, vocab = load_cloth_review_data()

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
y_train = y_train.astype(np.int32)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)
y_test = y_test.astype(np.int32)

pretrained_model_path = 'models/base_0.h5'
pretrained_model = load_model(pretrained_model_path)

def convert_pourcentage_to_quantity(pourcentage):
    return int(pourcentage * len(X_train) / 100)

def select_samples_randomly(quantity):
    # Select samples randomly
    random_indices = np.random.choice(len(X_train), quantity, replace=False)
    X_train_random = X_train[random_indices]
    y_train_random = y_train[random_indices]
    return X_train_random, y_train_random

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

def retrain_model(X_train_selected, y_train_selected, algorithm, pourcentage):

    model_path = 'models/base_0.h5'
    model = load_model(model_path)
    # checkpointer = ModelCheckpoint(f"models/{algorithm}_{pourcentage}.h5", save_best_only=True, verbose=1)
    model.fit(X_train_selected, y_train_selected, epochs=epochs, batch_size=batch_size)
    model.save(f"models/{algorithm}_{pourcentage}.h5")

def evaluate_model(algorithm, pourcentage):
    model_path = f'models/{algorithm}_{pourcentage}.h5'
    model = load_model(model_path)
    
    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
    result = f'{algorithm} - {pourcentage} - {acc:.4f}'
    print(result)
    return result


algorithms = {
    # "random": select_samples_randomly,
    "least_confidence": select_samples_by_least_confidence,
    "margin": select_samples_by_margin,
    "entropy": select_samples_by_entropy
}

pourcentages = [
    10, 
    15, 
    20, 
    25, 
    30, 
    35,
]

def main():
    results = []
    results.append("Model - Pourcentage - Accuracy")
    results.append("-------------------------------")
    results.append(evaluate_model("base", 0))
    for algorithm in algorithms:
        for pourcentage in pourcentages:
            quantity = convert_pourcentage_to_quantity(pourcentage)
            X_train_selected, y_train_selected = algorithms[algorithm](quantity)
            retrain_model(X_train_selected, y_train_selected, algorithm, pourcentage)
            results.append(evaluate_model(algorithm, pourcentage))
    print(results)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'results/{today}', "w") as fichier:
    # Ajouter un retour à la ligne après chaque élément de la liste
        fichier.writelines(ligne + "\n" for ligne in results)


if __name__ == "__main__":
    main()
