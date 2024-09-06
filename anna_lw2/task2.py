import random as rd
import numpy as np

def generateData(amount):
    return np.random.rand(amount, 2)

class Perceptron:
    def __init__(self):
        self.w = np.array([rd.random(), rd.random()])   # Веса
        self.w /= sum(map(lambda x: x * x, self.w)) ** 0.5   # Нормализуем веса
        self.bias = rd.random()     # Порог
        self.n = 4      # Количество итераций
        self.nu = 0.5   # Скорость обучения

    # Функция предсказания
    def predict(self, data):
        return np.where((np.dot(data, self.w) + self.bias) > 0, 1, -1)

    # Обучение
    def train(self, td, tda):
        for _ in range(self.n):
            for i, td_i in enumerate(td):
                tda_pred = self.predict(td_i) # Пытаемся предсказат результат
                self.w += self.nu * (tda[i] - tda_pred) * td_i # Исправляем веса на основе отклонения от реального ответа

class Adaline:
    def __init__(self):
        self.w = np.array([rd.random(), rd.random()])   # Веса
        self.bias = rd.random()
        self.w /= (sum(map(lambda x: x * x, self.w)) + self.bias ** 2) ** 0.5
        self.bias /= rd.random()
        self.n = 4   # Количество итераций
        self.nu = 0.5   # Скорость обучения

    # Функция предсказания
    def predict(self, data):
        return np.where((np.dot(data, self.w) + self.bias) > 0, 1, -1)

    # Обучение
    def train(self, td, tda):
        for _ in range(self.n):
            for i, td_i in enumerate(td):
                self.w += self.nu * (tda[i] - np.dot(td_i, self.w) + self.bias) * td_i # Исправляем веса

train_data = generateData(20)
train_data_answ = np.where(train_data[:, 0] > train_data[:, 1], 1, -1)

test_data = generateData(1000)
test_data_answ = np.where(test_data[:, 0] > test_data[:, 1], 1, -1)

nn_p = Perceptron()
nn_a = Adaline()

nn_p.train(train_data, train_data_answ)
nn_a.train(train_data, train_data_answ)

test_data_pred_p = nn_p.predict(test_data)
test_data_pred_a = nn_a.predict(test_data)

accuracy_p = np.mean(test_data_pred_p == test_data_answ)
accuracy_a = np.mean(test_data_pred_a == test_data_answ)

print(f"Точность перцептрона: {accuracy_p:.2f}")
print(f"Точность адалайна: {accuracy_a:.2f}")