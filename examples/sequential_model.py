import mnist
import numpy as np

from beacon.nn import beacon
from beacon.nn import Linear, BatchNorm
from beacon.nn.activations import ReLU, Softmax
from beacon.nn.models import Sequential
from beacon.optim import SGD
from beacon.data import data_generator
from beacon.functional import functions as F

model = Sequential(
    Linear(784, 64),
    BatchNorm(input_shape=(1, 64)),
    ReLU(),
    Linear(64,10),
    Softmax()
)
optimizer = SGD(model.parameters(), lr=0.01)

# Preparing training data
x_train = mnist.train_images().reshape(60000, 784) / 255.0
y_train = mnist.train_labels().reshape(60000, 1)
y_train = np.eye(10)[y_train].reshape(60000, 10)

# Preparing validation data
x_test = mnist.test_images().reshape(10000, 784) / 255.0
y_test = mnist.test_labels().reshape(10000, 1)

# Training
model.train()
for epoch in range(1, 6):
    full_loss = 0
    n_loss = 0
    for x, y in data_generator(x_train, y_train, batch_size=128, shuffle=True):
        optimizer.zero_grad()
        output = model(x)
        loss = F.categorical_crossentropy(output, y)
        loss.backward()
        optimizer.step(epoch)
        full_loss += loss.item()
        n_loss += 1
    print(f"Epoch: {epoch}, Loss: {full_loss/n_loss}")

# Evaluation
model.eval()
correct = 0
for x, y in zip(x_test, y_test):
    output = model(x)
    if (output.argmax() == y[0]):
        correct += 1
print(f"Accuracy: {correct/100}%")