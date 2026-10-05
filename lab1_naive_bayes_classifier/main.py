## libraries
import pandas as pd
import numpy as np

## naive bayes classifier class
class Nbayes:
    trained = False

    def fit(self, X_train, y_train):
        # a priori probabilities
        self.classes = np.unique(y_train)
        self.prior = {}
        for c in self.classes:
            self.prior[c] = np.sum(y_train == c) / len(y_train)
        
        # conditional probabilities
        self.cond_prob = {}
        for c in self.classes:
            self.cond_prob[c] = {}
            X_c = X_train[y_train == c]
            for col in X_train.columns:
                self.cond_prob[c][col] = {}
                for val in np.unique(X_train[col]):
                    self.cond_prob[c][col][val] = np.sum(X_c[col] == val) / len(X_c)
        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise ValueError
        y_predict = []
        for _, row in X_test.iterrows():
            posteriors = {}
            for c in self.classes:
                posteriors[c] = self.prior[c]
                for col in X_test.columns:
                    if row[col] in self.cond_prob[c][col]:
                        posteriors[c] *= self.cond_prob[c][col][row[col]]
            y_predict.append(max(posteriors, key=posteriors.get))
        return y_predict

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError
        y_predict = self.predict(X_test)
        # return accuracy, fraction of correct predictions:
        return (np.array(y_test) == np.array(y_predict)).sum()/len(y_test)

## main function
def main(case=1):
    # load the dataset
    if case == 0:
        data = pd.read_csv("weather/weather.data", sep=r'\s+', names=["outlook", "temperature", "humidity", "windy", "play"])
    elif case == 1:
        data = pd.read_csv("breast_cancer/breast-cancer.data", names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"])
    else:
        raise ValueError("Invalid case number. Please choose 0 for weather data or 1 for breast cancer data.")

    # split x and y
    if case == 0:
        X = data[["outlook", "temperature", "humidity", "windy"]]
        y = data["play"]
    elif case == 1:
        X = data[["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]]
        y = data["Class"]
    else:
        raise ValueError("Invalid case number. Please choose 0 for weather data or 1 for breast cancer data.")

    # create a new instance of the Naive Bayes classifier
    nb = Nbayes()

    # fit the model
    nb.fit(X, y)

    # test the model
    accuracy = nb.test(X, y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()