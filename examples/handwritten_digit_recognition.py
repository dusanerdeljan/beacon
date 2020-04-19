import mnist
import numpy as np

from beacon.nn import beacon
from beacon.nn import Module, Linear
from beacon.functional import functions as F
from beacon.optim import SGD, Adam
from beacon.data import data_generator

class Model(Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(inputs=784, outputs=64)
        self.fc2 = Linear(inputs=64, outputs=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

model = Model()
optimizer = Adam(model.parameters(), lr=0.01)

# Preparing training data
x_train = mnist.train_images().reshape(60000, 784) / 255.0
y_train = mnist.train_labels().reshape(60000, 1)
y_train = np.eye(10)[y_train].reshape(60000, 10)

# Preparing validation data
x_test = mnist.test_images().reshape(10000, 784) / 255.0
y_test = mnist.test_labels().reshape(10000, 1)

# Training
for epoch in range(1, 11):
    full_loss = 0
    n_loss = 0
    for x, y in data_generator(x_train, y_train, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        output = model(x)
        loss = F.categorical_crossentropy(output, y)
        loss.backward()
        optimizer.step(epoch)
        full_loss += loss.item()
        n_loss += 1
    print(f"Epoch: {epoch}, Loss: {full_loss/n_loss}")

# Evaluation
correct = 0
for x, y in zip(x_test, y_test):
    output = model(x)
    if (output.argmax() == y[0]):
        correct += 1
print(f"Accuracy: {correct/100}%")