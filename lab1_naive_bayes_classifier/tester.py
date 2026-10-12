import io
import re
import sys
import numpy as np

# Importa la funzione main dal tuo file esistente
from NbayesClassifier import main


def run_experiment(n_runs=100):
    accuracies = []

    for _ in range(n_runs):
        # Cattura lo stdout per leggere la stampa del main()
        buffer = io.StringIO()
        sys.stdout = buffer

        try:
            main()
        finally:
            sys.stdout = sys.__stdout__  # Ripristina lo stdout originale

        output = buffer.getvalue()

        # Estrae il valore numerico dell'accuratezza dall'output
        match = re.search(r"Accuratezza del modello:\s*([\d\.]+)%", output)
        if match:
            accuracies.append(float(match.group(1)))

    if accuracies:
        media = np.mean(accuracies)
        dev_std = np.std(accuracies)
        print(f"Esecuzioni completate: {len(accuracies)}/{n_runs}")
        print(f"Accuratezza media: {media:.2f}% (±{dev_std:.2f}%)")
    else:
        print("Nessun risultato estratto.")


if __name__ == "__main__":
    run_experiment(100)
