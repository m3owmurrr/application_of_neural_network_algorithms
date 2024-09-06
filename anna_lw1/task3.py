from random import random
from math import sqrt, exp

f = lambda x: 1/(1+exp(-x))

class Neuron:
    def __init__(self):
        self.nu = 0.5   #коэф обучения
        self.fr = self.nu*0.1   # коэфициент забывания
        self.w = [random(), random()]   # веса
        divider = sqrt(self.w[0]**2 + self.w[1]**2) # нормализуем веса
        self.w[0] /= divider
        self.w[1] /= divider

    def calculate(self, x):
            return self.w[0]*x[0] + self.w[1]*x[1]

    def recalculate(self, x, yi):
        self.w[0] = self.w[0]*(1-self.fr) + self.nu*x[0]*yi
        self.w[1] = self.w[1]*(1-self.fr) + self.nu*x[1]*yi


class NeuralNetwork:
    def __init__(self):
        self.x = [
            [0.97, 0.2],
            [1, 0],
            [-0.72, 0.7],
            [-0.67, 0.74],
            [-0.8, 0.6],
            [0, -1],
            [0.2, -0.97],
            [-0.3, -0.95],
        ]
        self.neurons = [Neuron() for i in range(2)]

    def __str__(self):
        s=''
        for neuron in self.neurons:
            s += str(neuron.w) + '\n'
        return s

    def start(self):
        u, y = 0, 0
        for i in range(len(self.x)):
            for j in range(2):
                u = self.neurons[j].calculate(self.x[i])    # Суммируем входные взвешенные данные
                y = f(u)                                    # Логическая функция активации, вычислям выходное значени
                self.neurons[j].recalculate(self.x[i], y)   # Изменяем веса на основе входного и выходного сигналов (Осовский)
                # print(self.neurons[j].w)


nn = NeuralNetwork()
print(nn)
nn.start()
print(nn)