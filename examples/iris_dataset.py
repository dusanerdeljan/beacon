import csv
import numpy as np
from random import shuffle
from collections import OrderedDict

from beacon.nn import beacon
from beacon.nn.models import Sequential
from beacon.nn import Linear
from beacon.nn.activations import ReLU, Softmax
from beacon.optim import Adam, SGD
from beacon.functional import functions as F
from beacon.data import data_generator


classes = OrderedDict({'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2})

def create_model():
    model = Sequential(
        Linear(inputs=4, outputs=8),
        ReLU(),
        Linear(inputs=8, outputs=3),
        Softmax()
    )
    return model

def read_data():
    dataset = []
    max_row = np.zeros(shape=4, dtype=np.float)
    with open('datasets\\iris.csv', 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            max_row = np.maximum(max_row, np.array(row[:-1], dtype=np.float))
            dataset.append(row)
    shuffle(dataset)
    inputs = []
    labels = []
    for row in dataset:
        inputs.append(np.array(row[:-1], dtype=np.float))
        labels.append(classes[row[-1]])
    return np.array(inputs, dtype=np.float) / max_row, np.eye(3)[np.array(labels)]

def train(model, x_train, y_train):
    optimizer = SGD(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1, 201):
        full_loss = 0
        n_loss = 0
        for x, y in data_generator(x_train, y_train, batch_size=10, shuffle=True):
            optimizer.zero_grad()
            output = model(x)
            loss = F.categorical_crossentropy(output, y)
            loss.backward()
            full_loss += loss.item()
            optimizer.step(epoch)
            n_loss += 1
        print(f"Epoch: {epoch}, Loss: {full_loss/n_loss}")

def evaluate(model, x_test, y_test):
    model.eval()
    correct = 0
    for x, y in zip(x_test, y_test):
        output = model(x)
        output_idx = output.argmax()
        y_idx = np.argmax(y)
        print(f"Predicted: {list(classes.keys())[output_idx]}, Actual: {list(classes.keys())[y_idx]}")
        if (output.argmax() == np.argmax(y)):
            correct += 1
    print(f"Accuracy: {100*correct/x_test.shape[0]}%")

if __name__ == "__main__":
    model = create_model()
    x, y = read_data()
    test_split = int(0.7 * x.shape[0])
    x_train, y_train = x[:test_split], y[:test_split]
    x_test, y_test = x[test_split:], y[test_split:]
    train(model, x_train, y_train)
    evaluate(model, x_test, y_test)