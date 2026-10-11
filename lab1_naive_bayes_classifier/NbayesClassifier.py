import pandas as pd
import numpy as np


class DiscreteNaiveBayes:
    # Flag per tracciare lo stato di addestramento del modello
    trained = False

    def __init__(self, a=1.0):
        # Parametro per il Laplace smoothing
        self.a = a

    def fit(self, X_train, Y_train):
        # Identifica le classi uniche nel target
        self.classes = np.unique(Y_train)
        self.prior = {}

        # Calcola le probabilità a priori per ciascuna classe
        for c in self.classes:
            self.prior[c] = np.sum(Y_train == c) / len(Y_train)

        self.cond_prob = {}
        
        # Calcola le probabilità condizionate
        for c in self.classes:
            self.cond_prob[c] = {}
            # Isola i dati appartenenti alla classe corrente
            X_c = X_train[Y_train == c]

            for col in X_train.columns:
                self.cond_prob[c][col] = {}
                n_unique_values = len(np.unique(X_train[col]))

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
        for _, row in X_test.iterrows():
            posteriors = {}

            # Calcola la probabilità a posteriori per ciascuna classe
            for c in self.classes:
                # Usa il logaritmo per evitare underflow numerico
                posterior = np.log(self.prior[c])

                for col in X_test.columns:
                    v = len(np.unique(X_test[col])) if col in X_test.columns else 1
                    
                    # Recupera la probabilità o ne stima una per valori non visti
                    if row[col] in self.cond_prob[c][col]:
                        prob = self.cond_prob[c][col][row[col]]
                    else:
                        prob = self.a / (len(self.cond_prob[c][col]) + self.a * v)

                    posterior += np.log(prob)

                posteriors[c] = posterior

            # Sceglie la classe con il valore massimo
            y_predict.append(max(posteriors, key=posteriors.get))

        return y_predict

    def test(self, X_test, Y_test):
        if not self.trained:
            raise ValueError("Modello non addestrato.")

        # Genera predizioni e ne valuta l'accuratezza rispetto ai valori reali
        y_predict = self.predict(X_test)
        accuracy = (np.array(Y_test) == np.array(y_predict)).sum() / len(Y_test)
        return accuracy


def read(file, hasHeader=False, separator=',', na_values='?'):
    # Importa i dati forzandoli a stringa
    data = pd.read_csv(
        file,
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

    return data_clean, summary_df


def main():
    # Dataset di riferimento: 
    #   Breast Cancer -> 0
    #   Weather -> 1
    esempio = 1

    # Parametri operativi
    if esempio == 0:
        file_path = "breast_cancer/breast-cancer.data"
        target_col_index = 0
        hasHeader = False
        separator = ','
        na_values = '?'
        a_smoothing = 1.0
        view_summary = True
        
    elif esempio == 1:
        file_path = "weather/weather.data"
        target_col_index = -1
        hasHeader = True
        separator = r'\s+'
        na_values = '?'
        a_smoothing = 1.0
        view_summary = True
    else:
        raise ValueError("Esempio non valido. Scegliere 0 o 1.")

    # Lettura ed estrazione del dataset e del riassunto
    data, summary = read(file_path, hasHeader=hasHeader, separator=separator, na_values=na_values)

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
    model = DiscreteNaiveBayes(a=a_smoothing)
    model.fit(X, y)

    accuracy = model.test(X, y)
    print(f"Accuratezza del modello: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main() 