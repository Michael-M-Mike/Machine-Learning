import csv


def get_data_from_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        table = list(reader)
        return table[0], table[1:]


def labels_frequency(data_table):

    labels_dict = {}
    for row in data_table:

        label = row[-1]
        if label not in labels_dict.keys():
            labels_dict[label] = 1
        else:
            labels_dict[label] += 1

    return labels_dict


def majority(d):
    for key in d:
        d[key] = float(d[key].strip("%"))
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def unique_values(data_table, feature):
    index = header.index(feature)
    return set([row[index] for row in data_table])


def index_of(element, list):
    return list.index(element)


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def get_info_gain(true_node, false_node, current_uncertainty):

    p = float(len(true_node.data)) / (len(true_node.data) + len(false_node.data))
    return current_uncertainty - p * true_node.impurity() - (1 - p) * false_node.impurity()


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        if is_numeric(self.value):
            condition = ">="
        else:
            condition = "=="

        return f"Is {header[self.column]} {condition} {str(self.value)}?"

    def ask(self, row):
        val = row[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value


class Node:

    def __init__(self, data):
        self.data = data
        self.true_node = None
        self.false_node = None

        self.question = None

    def prediction(self):

        prediction_dict = labels_frequency(self.data)
        total = sum(prediction_dict.values())

        for key in prediction_dict.keys():
            prediction_dict[key] = str(round((prediction_dict[key] * 100) / total, 2)) + "%"

        return prediction_dict

    def impurity(self):
        labels_dict = labels_frequency(self.data)
        impurity = 1

        for label in labels_dict:
            probability_of_label = labels_dict[label] / float(len(self.data))
            impurity -= probability_of_label ** 2

        return impurity

    def split(self, question):
        true_rows, false_rows = [], []

        for row in self.data:
            if question.ask(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        return true_rows, false_rows

    def split_gain_question(self):

        best_gain = 0
        best_question = None

        n_features = len(header) - 1

        for col in range(n_features):

            # review text
            if col == 25:
                continue

            values = set([row[col] for row in self.data])
            for val in values:

                question = Question(col, val)
                true_rows, false_rows = self.split(question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = get_info_gain(Node(true_rows), Node(false_rows), self.impurity())

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def populate(self):

        gain, question = self.split_gain_question()

        if gain < 0.001:
            return

        true_rows, false_rows = self.split(question)

        self.true_node = Node(true_rows)
        self.false_node = Node(false_rows)
        self.question = question

        self.true_node.populate()
        self.false_node.populate()

    def classify(self, example):

        if self.true_node is None and self.false_node is None:
            return self.prediction()

        if self.question.ask(example):
            return self.true_node.classify(example)
        else:
            return self.false_node.classify(example)


class DecisionTreeClassifier:

    def __init__(self, data):
        self.root = Node(data)

    def train(self):
        self.root.populate()

    def classify(self, example):
        return self.root.classify(example)


header, training_data = get_data_from_csv("sample_train.csv")
development_data = get_data_from_csv("sample_dev.csv")[1]
testing_data = get_data_from_csv("sample_test.csv")[1]

print("Training Started...")
t = DecisionTreeClassifier(training_data)
t.train()
print("Training Complete.")


print("\nClassifying the development dataset...")
correct = 0
incorrect = 0
for row in development_data:

    d = t.classify(row)

    expected = row[-1]
    actual = majority(d)

    if expected == actual:
        #print("Correct")
        correct += 1
    else:
        #print("Incorrect")
        incorrect += 1
print("Classification Complete.")
total = correct + incorrect
print(f"Total Samples = {total}")
print(f"Correct = {correct}")
print(f"Incorrect = {incorrect}")
print(f"Accuracy = {round(correct * 100 / total, 2)}%")


print("\nClassifying the testing dataset...")
with open("output.txt", "w") as text_file:
    for row in testing_data:
        d = t.classify(row)
        prediction = majority(d)
        text_file.write(f"{prediction}\n")
print("Classification Complete.")

for i in range(2):
    print("\nEnter a sample to classify:")
    sample = []
    for feature in header[:-2]:
        x = input(f"{feature}: ")
        sample.append(x)
    print(f"Prediction: {t.classify(sample)}")
