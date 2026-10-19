from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# kNN classifier
class Knn:
    # constructor
    def __init__(self, k=3, batch_size=128):
        if k < 1:
            raise ValueError("k must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.X_train_flattened = None
        self.y_train = None
        self.isTrained = False

    # fit method
    def fit(self, X, y):
        self.X_train = X
        self.X_train_flattened = X.reshape(X.shape[0], -1).astype(np.float32)
        self.y_train = y
        self.isTrained = True

    # predict method
    def predict(self, X):
        if not self.isTrained:
            raise Exception("Model is not trained yet. Call fit() before predict().")

        if self.k > len(self.X_train):
            raise ValueError("k cannot be greater than the number of training samples")

        # Avoid allocating the full test-by-training distance matrix. For MNIST
        # that matrix is large enough to make the operating system kill Python.
        X_flattened = X.reshape(X.shape[0], -1).astype(np.float32)
        X_train_flattened = self.X_train_flattened
        train_squared_norms = np.sum(X_train_flattened ** 2, axis=1)

        print("Flattened images")

        predictions = []
        for start in range(0, len(X_flattened), self.batch_size):
            batch = X_flattened[start:start + self.batch_size]

            # ||a-b||² = ||a||² + ||b||² - 2a·b; only one batch is held in memory.
            distances = (
                np.sum(batch ** 2, axis=1, keepdims=True)
                + train_squared_norms
                - 2 * batch @ X_train_flattened.T
            )
            nearest_indices = np.argpartition(
                distances, self.k - 1, axis=1
            )[:, :self.k]

            for labels in self.y_train[nearest_indices]:
                unique, counts = np.unique(labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])

            print(f"Processed {min(start + self.batch_size, len(X_flattened))}/"
                  f"{len(X_flattened)} images")

        print("Predictions made")

        return np.asarray(predictions)


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # create a kNN classifier
    knn = Knn(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    # calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()