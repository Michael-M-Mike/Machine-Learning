import csv
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


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


class Model:

    def __init__(self):
      self.weights = []

    def fit(self, training_data, learning_rate, iterations):
      print("Training...")
      self.weights = gradient_descent(training_data, learning_rate, iterations)
      print("Training Complete.")


    def predict(self, testing_data):

      y = []
      h = []

      print("Testing...")
      for example in testing_data:
        prediction = hypothesis(self.weights, example.features)

        print(f"Prediction: {prediction}\t\t Actual: {example.label}")

        h.append(prediction)
        y.append(example.label)

      print("Testing Complete.")
      print(f"Testing Loss = {cost(h, y, len(y))}")


def feature_list(data, feature_index):
    return_list = []
    for ex in data:
        return_list.append(ex.features[feature_index])

    return return_list


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


    # Plot variables
    c = []
    iterations = [i for i in range(num_iterations)]

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

        c.append(cost(h, y, m))

    plt.plot(iterations, c, '-')
    plt.title("Cost Function")
    plt.show()

    return w


def wout_sklearn():

    model = Model()
    model.fit(training_data, 0.01, 500)
    print("Coefficients:")
    print(model.weights)
    print()
    model.predict(testing_data)


def w_sklearn():
    
    model = linear_model.LinearRegression()

    X_train = [x.features for x in training_data]
    y_train = [y.label for y in training_data]

    X_test = [x.features for x in testing_data]
    y_test = [y.label for y in testing_data]

    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)


    print("Testing...")
    for i in range(len(y_predictions)):
      print(f"Prediction: {y_predictions[i]}\t\t Actual: {y_test[i]}")

    print("Testing Complete.")

    # The coefficients
    print('Coefficients: \n', model.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_predictions))
    

def main():
  data_set = Data("Admission.csv")
  data_set.scale_features()

  training_data, testing_data = data_set.train_test_split(0.01)

  print("Linear Regression without Sklearn")
  wout_sklearn()

  print("\nLinear Regression with Sklearn")
  w_sklearn()


main()
