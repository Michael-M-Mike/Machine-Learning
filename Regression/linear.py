import csv
import numpy as np
from random import shuffle


def feature_list(data, feature_index):
    return_list = []
    for ex in data:
        return_list.append(ex.features[feature_index])

    return return_list


class Example:

    def __init__(self, features, label):
        self.features = features
        self.label = label

    def __repr__(self):
        return f"Features: {self.features} \t\t Label: {self.label}"


class Data:

    def __init__(self, csv_file_path=""):

        with open(csv_file_path) as file:
            reader = csv.reader(file)

            # Skip data set header row
            non_features = 2
            self.header = next(reader)
            self.num_features = len(self.header) - non_features

            self.data = []
            for row in reader:
                self.data.append(
                    Example(
                        [float(feature) for feature in row[1:-1]],
                        float(row[-1])
                    )
                )

            self.num_examples = len(self.data)

    def train_test_split(self, testing_data_percentage):

        holdout = int(testing_data_percentage * self.num_examples)
        shuffle(self.data)

        testing = self.data[:holdout]
        training = self.data[holdout:]

        return training, testing

    def scale_features(self):

        maximum = []
        minimum = []
        average = []

        for i in range(self.num_features):
            maximum.append(max(feature_list(self.data, i)))
            minimum.append(min(feature_list(self.data, i)))
            average.append(sum(feature_list(self.data, i)) / len(self.data))

        scaled = []
        for ex in self.data:
            scaled.append(
                Example(
                    [round((ex.features[i] - average[i]) / (maximum[i] - minimum[i]) * 10, 2)
                     for i in range(len(ex.features))], ex.label
                )
            )

        self.data = scaled


def cost(h, y, m):
    return (1 / (2 * m)) * np.sum(np.square(np.array(h) - np.array(y)))


def hypothesis(w, x):

    """
    :param w: (list) current weight parameters
    :param x: (list) inputs for a given example
    :return: (float) the hypothesis for the given example
    """

    x_with_bias = [1]
    for element in x:
        x_with_bias.append(element)

    h = np.dot(w, x_with_bias)
    return round(h, 2)


def partial_derivative(h, y, x, m):
    return (1 / m) * np.sum((h - y) * x)


def gradient_descent(data, alpha, num_iterations):

    # Start with null weights
    m = len(data)
    w = [0 for i in range(len(data[0].features) + 1)]

    for i in range(num_iterations):

        # Obtain predictions with current weights
        h = []
        y = []
        for ex in data:
            h.append(hypothesis(w, ex.features))
            y.append(ex.label)

        # Update weights
        for i in range(len(w)):
            if i == 0:
                x = [1 for j in range(m)]

            else:
                x = feature_list(data, i - 1)

            w[i] += -alpha * partial_derivative(np.array(h), y, x, m)

    return w


def main():
    data_set = Data("Admission.csv")
    data_set.scale_features()

    training_data, testing_data = data_set.train_test_split(0.3)

    print("Training...")
    weights = gradient_descent(training_data, 0.05, 1000)
    print("Training Complete.")

    print("Testing...")
    h = []
    y = []
    for example in testing_data:
        prediction = hypothesis(weights, example.features)
        h.append(prediction)
        y.append(example.label)

        print(f"Prediction: {prediction}\t\tActual: {example.label}")
    print("Testing Complete.")
    print(f"Testing Loss = {cost(h, y, len(y))}")


if __name__ == '__main__':
    main()
