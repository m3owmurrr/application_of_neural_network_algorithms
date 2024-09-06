import numpy as np
import random as rd


class NeuronNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.weights_input_to_hidden = np.random.rand(input_layer_size, hidden_layer_size)
        self.weights_hidden_to_output = np.random.rand(hidden_layer_size, output_layer_size)

        self.bias_hidden = np.random.rand(hidden_layer_size)
        self.bias_output = np.random.rand(output_layer_size)
    def activation_func(self, x):
        return np.tanh(x)
    def derivator(self, x):
        return 1 - np.tanh(x) ** 2
    def forward(self, X):
        self.input_hidden = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.activation_func(self.input_hidden)
        self.output = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        return self.activation_func(self.output)
    def backward(self, X, y, output, learning_rate):
        error = y - output
        output_delta = error * self.derivator(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T)
        hidden_delta = hidden_error * self.derivator(self.hidden_output)

        self.weights_hidden_to_output += np.outer(self.hidden_output.T, output_delta) * learning_rate
        self.weights_input_to_hidden += np.outer(X.T, hidden_delta) * learning_rate
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i, x_i in enumerate(X):
                output = self.forward(x_i)
                self.backward(x_i, y[i], output, learning_rate)
            if epoch % 100 == 0:
                out = self.forward(X)
                loss = np.mean((y - out) ** 2)
                print(f'{epoch} {loss}')


x_train = np.array([
    # X
    np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]),
    np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]),
    np.array([
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1]
    ]),
    # Y
    np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    np.array([
        [1, 0, 1],
        [0, 0, 0],
        [0, 1, 0]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0]
    ]),
    # I
    np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0]
    ]),
    # L
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]),
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ]),
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1]
    ]),
    np.array([
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 1]
    ]),
    np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
])
y_train = []
for i in [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]:
    t = [0] * 4
    t[i] = 1
    y_train.append(t)
y_train = np.array(y_train)
x_test = np.array([
    # X
    np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]),
    np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1]
    ]),
    # Y
    np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    np.array([
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    # I
    np.array([
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 0]
    ]),
    np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ]),
    # L
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1]
    ]),
    np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]),
])
y_test = []
for i in [3, 3, 2, 2, 1, 1, 0, 0]:
    t = [0] * 4
    t[i] = 1
    y_test.append(t)
y_test = np.array(y_test)

X_train = np.array([i.flatten() for i in x_train])
X_test = np.array([i.flatten() for i in x_test])

nn = NeuronNetwork(9, 6, 4)
nn.train(X_train, y_train, 1000, 0.005)

c = []
for yp, y in zip(nn.forward(X_test), y_test):
    c.append(yp.argmax() == y.argmax())
print(c)
print(f'Точность: {c.count(True)/len(c)}')
