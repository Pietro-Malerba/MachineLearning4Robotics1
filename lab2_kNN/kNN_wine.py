from sklearn.datasets import load_wine
import numpy as np
from scipy import stats
import time


class Knn:

    def __init__(self, _k):
        self.k = _k
        self.X = None
        self.y = None
        self.min_X = None
        self.max_X = None

    def fit(self, X_train, y_train):
        # normalise the training data
        self.min_X = np.min(X_train, axis=0)
        self.max_X = np.max(X_train, axis=0)
        X_normalised = (X_train - self.min_X) / (self.max_X - self.min_X)
        self.X = X_normalised
        self.y = y_train

    def predict(self, X_test):
        # Apply the same normalisation used for the training data.
        X = (X_test - self.min_X) / (self.max_X - self.min_X)

        # Euclidean distance calculation ((a-b)^2 = a^2 - 2ab + b^2)
        distances = (np.sum(self.X ** 2, axis=1, keepdims=True) - 2 * self.X @ X.T + np.sum(X ** 2, axis=1)[None, :])
        k_indices = np.argpartition(distances, self.k - 1, axis=0)[: self.k]
        k_labels = self.y[k_indices]

        # majority voting
        mode_result = stats.mode(k_labels, axis=0)
        predictions = mode_result.mode.flatten()

        return np.array(predictions)

    def test(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy




def main(k, n_train=100, n_test=78, random_state=5839759):

    if n_train + n_test > 178:
        raise ValueError("The sum of n_train and n_test must not exceed 178.")

    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Shuffle the dataset and split it into training and testing sets
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:n_train + n_test]], y[indices[n_train:n_train + n_test]]

    # create a kNN classifier
    knn = Knn(_k = k)
    knn.fit(X_train, y_train)
    accuracy = knn.test(X_test, y_test)
    return accuracy


if __name__ == "__main__":
    start_time = time.perf_counter()
    print(f"Accuracy: {main(3):.2f}")
    elapsed_time = time.perf_counter() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")