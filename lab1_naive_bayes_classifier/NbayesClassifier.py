import pandas as pd
import numpy as np
import re
import warnings
from io import StringIO
import random


class Nbayes:
    # Flag per tracciare lo stato di addestramento del modello
    trained = False

    def __init__(self, a=1.0):
        # Parametro per il Laplace smoothing
        self.a = a

    def fit(self, X_train, Y_train, levels=None):
        # Identifica le classi uniche nel target
        self.classes = np.unique(Y_train)
        self.prior = {}

        # Calcola le probabilità a priori per ciascuna classe
        for c in self.classes:
            self.prior[c] = np.sum(Y_train == c) / len(Y_train)

        self.cond_prob = {}
        self.levels = list(levels) if levels is not None else [
            X_train[col].nunique() for col in X_train.columns
        ]
        if len(self.levels) != len(X_train.columns):
            raise ValueError("Il numero di livelli deve corrispondere agli attributi.")
        
        # Calcola le probabilità condizionate
        for c in self.classes:
            self.cond_prob[c] = {}
            # Isola i dati appartenenti alla classe corrente
            X_c = X_train[Y_train == c]

            for col in X_train.columns:
                self.cond_prob[c][col] = {}
                n_unique_values = self.levels[X_train.columns.get_loc(col)]

                # Calcola frequenze e applica il Laplace smoothing
                for val in np.unique(X_train[col]):
                    count_value = np.sum(X_c[col] == val)
                    self.cond_prob[c][col][val] = (
                        count_value + self.a
                    ) / (len(X_c) + self.a * n_unique_values)

        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError("Modello non addestrato.")

        y_predict = []

        # Itera sulle osservazioni per predire la classe
        for index, row in X_test.iterrows():
            if any(
                row[col] not in self.cond_prob[c][col]
                for col in X_test.columns
                for c in self.classes
            ):
                warnings.warn(
                    f"Pattern {index} scartato: contiene valori non presenti nel training set.",
                    UserWarning,
                )
                continue
            posteriors = {}

            # Calcola la probabilità a posteriori per ciascuna classe
            for c in self.classes:
                # Usa il logaritmo per evitare underflow numerico
                posterior = np.log(self.prior[c])

                for col in X_test.columns:
                    prob = self.cond_prob[c][col][row[col]]
                    posterior += np.log(prob)

                posteriors[c] = posterior

            # Sceglie la classe con il valore massimo
            y_predict.append(max(posteriors, key=posteriors.get))

        return y_predict

    def test(self, X_test, Y_test):
        if not self.trained:
            raise ValueError("Modello non addestrato.")

        # Genera predizioni e ne valuta l'accuratezza rispetto ai valori reali
        valid_indices = [
            index for index, row in X_test.iterrows()
            if all(
                row[col] in self.cond_prob[c][col]
                for col in X_test.columns
                for c in self.classes
            )
        ]
        y_predict = self.predict(X_test)
        y_valid = np.asarray(Y_test)[X_test.index.get_indexer(valid_indices)]
        if len(y_valid) == 0:
            raise ValueError("Nessun pattern valido nel test set.")
        accuracy = (y_valid == np.array(y_predict)).sum() / len(y_valid)
        return accuracy


def read(file, hasHeader=False, separator=',', na_values='?'):
    with open(file, encoding='utf-8') as source:
        lines = source.readlines()

    levels = None
    if lines and re.fullmatch(r'\s*\[[^\]]*\]\s*', lines[0]):
        metadata = lines.pop(0).strip()
        levels = [int(value.strip()) for value in metadata[1:-1].split(',')]

    # Importa i dati forzandoli a stringa
    data = pd.read_csv(
        StringIO(''.join(lines)),
        sep=separator,
        engine='python',
        header=0 if hasHeader else None,
        dtype=str,
    )

    # Rimuove gli spazi e sostituisce i placeholder con NaN
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
    data = data.replace(na_values, np.nan)

    # Rimuove le righe con valori mancanti
    data_clean = data.dropna().reset_index(drop=True)

    # Crea un dataframe riassuntivo con statistiche sulle colonne
    summary_df = pd.DataFrame(
        {
            'Column': data_clean.columns,
            'Missing Values Removed': [data[col].isna().sum() for col in data.columns],
            'v_Unique_Count': [data_clean[col].nunique() for col in data_clean.columns],
            'Unique Values': [data_clean[col].unique() for col in data_clean.columns],
        }
    )

    feature_count = data_clean.shape[1] - 1
    if levels is None:
        levels = [data_clean.iloc[:, col].nunique() for col in range(feature_count)]
    if len(levels) != feature_count:
        raise ValueError("La riga dei livelli deve descrivere tutti gli attributi.")

    return data_clean, summary_df, levels


def binarize_iris(train, test, target_col_index=-1, use_median=False):
    # Calcola l'indice della colonna target se è negativo
    if target_col_index < 0:
        target_col_index = train.shape[1] + target_col_index

    # Verifica che il dataset abbia quattro attributi e una classe
    if train.shape[1] != 5 or test.shape[1] != 5:
        raise ValueError("Iris deve contenere quattro attributi e una classe.")

    train_binarized = train.copy()
    test_binarized = test.copy()
    thresholds = {}

    # Seleziona le colonne degli attributi da binarizzare
    feature_columns = [
        column for position, column in enumerate(train.columns)
        if position != target_col_index
    ]
    for column in feature_columns:
        # Converte i valori della colonna in numeri e calcola la soglia
        train_values = pd.to_numeric(train[column], errors='raise')
        test_values = pd.to_numeric(test[column], errors='raise')
        threshold = (
            train_values.median() if use_median else train_values.mean()
        )
        thresholds[column] = threshold

        # Trasforma i valori in 1 se superano la soglia, altrimenti in 0
        train_binarized[column] = (train_values > threshold).astype(int)
        test_binarized[column] = (test_values > threshold).astype(int)

    return train_binarized, test_binarized, thresholds


def main():
    # Dataset di riferimento: 
    #   Breast Cancer -> 0
    #   Weather -> 1
    #   Iris binarizzato -> 2
    esempio = 2
    view_summary = False

    # Parametri operativi
    if esempio == 0:
        file_path = "breast_cancer/breast-cancer.data"
        target_col_index = 0
        hasHeader = False
        separator = ','
        na_values = '?'
        a_smoothing = 1.0
    elif esempio == 1:
        file_path = "weather/weather.data"
        target_col_index = -1
        hasHeader = True
        separator = r'\s+'
        na_values = '?'
        a_smoothing = 1.0
    elif esempio == 2:
        file_path = "iris/iris.data"
        target_col_index = -1
        hasHeader = False
        separator = ','
        na_values = '?'
        a_smoothing = 1.0
    else:
        raise ValueError("Esempio non valido. Scegliere 0, 1 o 2.")

    # Lettura ed estrazione del dataset e del riassunto
    data, summary, levels = read(
        file_path, hasHeader=hasHeader, separator=separator, na_values=na_values
    )

    if view_summary:
        print("Summary Informativo:")
        print(summary)

    # Corregge l'indice del target se negativo
    if target_col_index < 0:
        target_col_index = data.shape[1] + target_col_index

    # Separa target (y) e features (X)
    y = data.iloc[:, target_col_index]
    X = data.drop(data.columns[target_col_index], axis=1)

    # Crea, addestra e testa il classificatore
    model = Nbayes(a=a_smoothing)
    train = data.sample(frac=0.8, random_state=random.randint(0, 10000)) 
    test = data.drop(train.index)

    if esempio == 2:
        train, test, thresholds = binarize_iris(
            train,
            test,
            target_col_index=target_col_index,
        )
        levels = [2] * (train.shape[1] - 1)
        if view_summary:
            print("Soglie usate per la binarizzazione:")
            print(thresholds)

    X_train = train.drop(train.columns[target_col_index], axis=1)
    y_train = train.iloc[:, target_col_index]
    X_test = test.drop(test.columns[target_col_index], axis=1)
    y_test = test.iloc[:, target_col_index]

    model.fit(X_train, y_train, levels=levels)

    accuracy = model.test(X_test, y_test)
    print(f"Accuratezza del modello: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main() 