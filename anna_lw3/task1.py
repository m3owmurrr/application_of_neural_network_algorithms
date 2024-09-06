import numpy as np
import matplotlib.pyplot as plt

# Функция Гаусса
# h = lambda x, c, r: np.exp(-1 * (((np.sqrt(sum((xi - ci)**2 for xi, ci in zip(x, c)))) ** 2) / (r ** 2)))
h = lambda x, c, r: np.exp(-((np.linalg.norm(x - c) ** 2) / (r ** 2)))


# centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
centers = np.array([ 0.88186153, -1.27463329,  1.04557055,  1.35683792, -0.07740024,]
)
# centers = np.random.uniform(-2.0, 2.0, 5)
print(centers)

x_train = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
y_train = np.array([-0.48, -0.78, -0.83, -0.67, -0.20, 0.70, 1.48, 1.17, 0.20])

class Radial_Neural_Network:
    def __init__(self, centers, radius, neurons_num):
        self.centers = centers
        self.radius = radius
        self.neurons_num = neurons_num

    def weights_init(self, x_train, y_train):
        # Формируем марицу H для подсчета весов
        H = np.zeros((len(x_train), self.neurons_num))
        for i, x in enumerate(x_train):
            for j, c in enumerate(self.centers):
                H[i, j] = h(x, c, self.radius)

        # Формируем вектор весов
        self.weights = np.linalg.inv(H.T @ H) @ H.T @ y_train

    def predict(self, x):
        output = 0
        for i in range(self.neurons_num):
            output += self.weights[i] * h(x, self.centers[i], self.radius)
        return output


rnn = Radial_Neural_Network(centers,  0.8, 5)
rnn.weights_init(x_train, y_train)

x_test = np.linspace(-2, 2, 100)
y_predict = [rnn.predict(x) for x in x_test]

plt.plot(x_test, y_predict, color='b')
plt.scatter(x_train, y_train, color='r')
plt.show()

# [ 0.88186153 -1.27463329  1.04557055  1.35683792 -0.07740024]
