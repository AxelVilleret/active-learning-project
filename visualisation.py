from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = 'results/results.json'

def update_json(file_path, method_name, percentage, performance, loss, history):
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

    for method in data.get('methodes', []):
        if method['nom'] == method_name:
            method['performances'].append(
                {
                    'pourcentage': percentage,
                    'accuracy_test': performance,
                    'loss_test': loss,
                    'accuracy_train': history['accuracy'] if history else [],
                    'loss_train': history['loss'] if history else [],
                }
            )
            break
    else:
        data['methodes'].append(
            {
                'nom': method_name, 
                'performances': 
                [
                    {
                        'pourcentage': percentage, 
                        'accuracy_test': performance,
                        'loss_test': loss,
                        'accuracy_train': history['accuracy'] if history else [],
                        'loss_train': history['loss'] if history else [],

                    }
                ]
            }
        )

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def plot_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for method in data['methodes']:
        if method['nom'] == 'base':
            # Pour la méthode 'base', tracez une ligne horizontale à la valeur de performance
            y_value = method['performances'][0]['accuracy_test']
            plt.axhline(y=y_value, label='base', color='r')
        else:
            performances = defaultdict(list)
            for perf in method['performances']:
                performances[perf['pourcentage']].append(perf['accuracy_test'])

            x = sorted(performances.keys())
            y = [sum(performances[p])/len(performances[p]) for p in x]

            plt.plot(x, y, label=method['nom'])

    plt.xlabel('Pourcentage')
    plt.ylabel('Performance de test')
    plt.legend()
    plt.show()


def main():
    # Appelez la fonction plot_graph avec le chemin du fichier
    plot_graph(RESULTS_PATH)

if __name__ == '__main__':
    main()
