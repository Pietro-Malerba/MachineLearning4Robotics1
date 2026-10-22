from keras.datasets import mnist
import numpy as np
from scipy import stats
import time


class Knn:

    def __init__(self, k, batch_size=128):
        self.k = k
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # flatten the training data
        self.X_train = X.reshape(len(X), -1).astype(np.float32)
        self.y_train = y
        print("Training data fitted successfully.")

    def predict(self, X):
        # flattening the test set
        X_test = X.reshape(len(X), -1).astype(np.float32)
        predictions = np.zeros(len(X_test), dtype=self.y_train.dtype)

        # introducing batch processing due to the dimentions of the dataset
        for i in range(0, len(X_test), self.batch_size):
            X_batch = X_test[i : i + self.batch_size]

            # Euclidean distance calculation ((a-b)^2 = a^2 - 2ab + b^2)
            distances = (np.sum(self.X_train ** 2, axis=1, keepdims=True) - 2 * self.X_train @ X_batch.T + np.sum(X_batch ** 2, axis=1)[None, :])
            k_indices = np.argpartition(distances, self.k - 1, axis=0)[: self.k]
            k_labels = self.y_train[k_indices]

            # majority voting
            mode_result = stats.mode(k_labels, axis=0)
            batch_predictions = mode_result.mode.flatten()
            predictions[i : i + len(X_batch)] = batch_predictions

            total_batches = (len(X_test) + self.batch_size - 1) // self.batch_size
            print(f"Processed batch {i // self.batch_size + 1} of {total_batches}", end='\r')

        print()  # Print a newline after the progress indicator
        return np.array(predictions)



def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # create a kNN classifier
    knn = Knn(k=3, batch_size=128)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    # calculate the accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")