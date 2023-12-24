from preprocess import load_cloth_review_data, load_food_review_data
import numpy as np
from keras.models import load_model, clone_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from visualization import update_json
import pandas as pd
from global_variables import *
from config import *
from utils import convert_pourcentage_to_quantity

selected_samples = {}

def retrain_model(X_train_selected, y_train_selected, X_validation, y_validation, algorithm, pourcentage):
    print(f"Retraining model with this number of samples: {len(X_train_selected)}")
    model = load_model(PRETRAINED_MODEL_PATH)
    checkpointer = ModelCheckpoint(f"models/{algorithm}_{pourcentage}.h5", save_best_only=True, verbose=1)
    history = model.fit(X_train_selected, y_train_selected, epochs=EPOCHS, validation_data=(
        X_validation, y_validation), batch_size=BATCH_SIZE, callbacks=[checkpointer])
    # model.save(f"models/{algorithm}_{pourcentage}.h5")
    return history

def evaluate_model(algorithm, pourcentage, X_test, y_test):
    model_path = f'models/{algorithm}_{pourcentage}.h5'
    model = load_model(model_path)
    
    # Evaluate the model on the test data
    loss, acc = model.evaluate(X_test, y_test)
    result = f'{algorithm} - {pourcentage} - {acc:.4f}'
    print(result)
    return loss, acc

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
    X_train, X_validation, X_test, y_train, y_validation, y_test, vocab = load_cloth_review_data()
    X_train = sequence.pad_sequences(X_train, maxlen=SEQUENCE_LENGTH)
    y_train = y_train.astype(np.int32)
    X_validation = sequence.pad_sequences(X_validation, maxlen=SEQUENCE_LENGTH)
    y_validation = y_validation.astype(np.int32)
    X_test = sequence.pad_sequences(X_test, maxlen=SEQUENCE_LENGTH)
    y_test = y_test.astype(np.int32)
    base_loss, base_acc = evaluate_model(BASE, 0, X_test, y_test)
    update_json(RESULTS_PATH, BASE, 0, base_acc, base_loss, None)
    for algorithm in algorithms:
        selected_samples[algorithm] = {}
        for pourcentage in pourcentages:
            quantity = convert_pourcentage_to_quantity(pourcentage, X_train)
            X_train_selected, y_train_selected = algorithms[algorithm](quantity, X_train, y_train)
            selected_samples[algorithm][pourcentage] = X_train_selected
            history = retrain_model(X_train_selected, y_train_selected, X_validation, y_validation, algorithm, pourcentage)
            loss, acc = evaluate_model(algorithm, pourcentage, X_test, y_test)
            update_json(RESULTS_PATH, algorithm, pourcentage, acc, loss, history.history)
    # in_common_pourcentages()

if __name__ == "__main__":
    main()
