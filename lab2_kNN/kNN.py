from keras.datasets import mnist
import numpy as np

# kNN classifier
class Knn:
    # constructor
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.isTrained = False

    # fit method
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.isTrained = True

    # predict method
    def predict(self, X):
        if not self.isTrained:
            raise Exception("Model is not trained yet. Call fit() before predict().")

        # flattening delle immagini in vettori
        X_flattened = X.reshape(X.shape[0], -1)
        X_train_flattened = self.X_train.reshape(self.X_train.shape[0], -1)

        # calcolo delle distanze euclidee tra i campioni di test e quelli di training
        distances = np.linalg.norm(X_flattened[:, np.newaxis] - X_train_flattened, axis=2)

        # selezione dei k vicini più vicini
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]

        # predizioni finali basate sul voto di maggioranza dei k nearest neighbours
        predictions = []
        for labels in k_nearest_labels:
            unique, counts = np.unique(labels, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            predictions.append(majority_vote)

        return predictions



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
