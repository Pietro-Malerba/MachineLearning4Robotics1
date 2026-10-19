from keras.datasets import mnist
import numpy as np

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
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim < 2:
            raise ValueError("X must contain at least one sample dimension")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if len(X) == 0:
            raise ValueError("the training set cannot be empty")

        self.X_train = X
        self.X_train_flattened = X.reshape(len(X), -1).astype(np.float32)
        self.y_train = y
        self.isTrained = True
        return self

    # predict method
    def predict(self, X):
        if not self.isTrained:
            raise Exception("Model is not trained yet. Call fit() before predict().")

        if self.k > len(self.X_train):
            raise ValueError("k cannot be greater than the number of training samples")

        X = np.asarray(X)
        if X.ndim < 2:
            raise ValueError("X must contain at least one sample dimension")
        X_flattened = X.reshape(len(X), -1).astype(np.float32)
        if X_flattened.shape[1] != self.X_train_flattened.shape[1]:
            raise ValueError("training and prediction samples have different shapes")

        predictions = []
        for start in range(0, len(X_flattened), self.batch_size):
            batch = X_flattened[start:start + self.batch_size]
            # Squared Euclidean distances, computed without Python loops.
            distances = (
                np.sum(batch ** 2, axis=1, keepdims=True)
                + np.sum(self.X_train_flattened ** 2, axis=1)
                - 2 * batch @ self.X_train_flattened.T
            )
            nearest = np.argpartition(distances, self.k - 1, axis=1)[:, :self.k]
            for indices in nearest:
                labels, counts = np.unique(self.y_train[indices], return_counts=True)
                predictions.append(labels[np.argmax(counts)])

            processed = min(start + len(batch), len(X_flattened))
            print(f"Processed {processed}/{len(X_flattened)} samples", end='\r')

        print()  # Print a newline after the progress indicator
        return np.asarray(predictions)


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # create a kNN classifier
    knn = Knn(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    # calculate accuracy
    accuracy = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()