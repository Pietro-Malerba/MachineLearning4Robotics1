import matplotlib.pyplot as plt
import numpy as np
import kNN_wine as knn

def main():
    k_values = [1,2,3,4,5,10,15,20,30,40,50]
    accuracies = []

    for k in k_values:
        acc = []
        for i in range(50):
            accuracy = knn.main(k, n_train=100, n_test=78, random_state=i)
            acc.append(accuracy)
        accuracies.append(acc)

    # Computing useful stats
    means = []
    stds = []
    for i in range(len(accuracies)):
        means.append(np.mean(accuracies[i]))
        stds.append(np.std(accuracies[i]))
        print(f"k = {k_values[i]} => {means[i]*100:.4f}% ± {stds[i]*100:.4f}%")

    # Plotting charts
    plt.bar([str(k) for k in k_values], means, yerr=stds, capsize=5)
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.title('kNN Classifier Accuracy for Different k Values')
    plt.show()


if __name__ == '__main__':
    main()