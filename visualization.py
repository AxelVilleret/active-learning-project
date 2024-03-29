from collections import defaultdict
import json
import os
from re import T
import matplotlib.pyplot as plt
from matplotlib.style import available
from global_variables import *
import numpy as np

available_methods = [BASE, RANDOM, CLUSTERING, REPRESENTATIVE_SAMPLING, LEAST_CONFIDENCE, MARGIN, ENTROPY, MIXED_WITH_LEAST_CONFIDENCE_AND_CLUSTERING, MIXED_WITH_MARGIN_AND_CLUSTERING]

def update_json(file_path, method_name, percentage, performance, loss, history):
    data = {
        METHODS: []
    }
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    performance = {
        PERCENTAGE: percentage,
        ACCURACY_TEST: performance,
        LOSS_TEST: loss,
        ACCURACY_TRAIN: history['accuracy'] if history else [],
        LOSS_TRAIN: history['loss'] if history else [],
    }

    for method in data.get(METHODS, []):
        if method[NAME] == method_name:
            method[PERFORMANCES].append(
                performance
            )
            break
    else:
        data[METHODS].append(
            {
                NAME: method_name, 
                PERFORMANCES: 
                [
                    performance
                ]
            }
        )

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def plot_graph(file_path=RESULTS_PATH, methods=available_methods, save_path=IMAGE_RESULTS_PATH, error_bar=True, scope='test'):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for method in data[METHODS]:
        if method[NAME] in methods:
            if method[NAME] == BASE:
                # Pour la méthode 'base', tracez une ligne horizontale à la valeur de performance
                y_value = method[PERFORMANCES][0][ACCURACY_TEST]
                plt.axhline(y=y_value, label=BASE, color='r')
            else:
                performances = defaultdict(list)
                for perf in method[PERFORMANCES]:
                    if scope == 'train':
                        performances[perf[PERCENTAGE]].append(perf[ACCURACY_TRAIN][-1])
                    elif scope == 'test':
                        performances[perf[PERCENTAGE]].append(perf[ACCURACY_TEST])

                x = sorted(performances.keys())
                y = [sum(performances[p])/len(performances[p]) for p in x]
                err = [np.std(performances[p]) for p in x]
                if error_bar:
                    plt.fill_between(x, np.array(y)-np.array(err), np.array(y)+np.array(err), alpha=0.2)

                plt.plot(x, y, label=method[NAME])

    plt.xlabel('Pourcentage')
    plt.ylabel('Performance de test')
    plt.legend()
    # plt.savefig(save_path)  # Enregistre le graphique dans un fichier
    plt.show()


def main():
    # Appelez la fonction plot_graph avec le chemin du fichier
    # plot_graph()
    # plot_graph(methods=[BASE, LEAST_CONFIDENCE, MARGIN], save_path='results/results_2.png')
    plot_graph(methods=[BASE, RANDOM, CLUSTERING, REPRESENTATIVE_SAMPLING, LEAST_CONFIDENCE, MARGIN, ENTROPY,
               MIXED_WITH_LEAST_CONFIDENCE_AND_CLUSTERING], save_path='results/results_3.png', error_bar=True)


if __name__ == '__main__':
    main()
