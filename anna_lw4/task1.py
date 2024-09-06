import numpy as np
import matplotlib.pyplot as plt

def normal(data):
    tdata = np.delete(data, 7, 1)
    for d in tdata:
        d[0] = 1 if d[0] == "M" else 0
        d[1] = 1 if d[1] == "Y" else 0
        for i in range(2, len(d)):
            d[i] = int(d[i]) / 100
    return tdata.astype(float)
class Kohonen_Neural_Network:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.3

        lower_bound = 0.5 - 1 / np.sqrt(self.input_size)
        upper_bound = 0.5 + 1 / np.sqrt(self.input_size)
        self.weights = np.random.uniform(lower_bound, upper_bound, (self.output_size, self.input_size))


    def train(self, train_data):
        R = np.zeros(self.output_size)
        while(round(self.learning_rate, 2) != 0.0):
            for i, d in enumerate(train_data):
                for j in range(len(R)):
                    R[j] = np.linalg.norm(d - self.weights[j])

                min_r = np.argmin(R)
                for j in range(self.input_size):
                    self.weights[min_r, j] = self.weights[min_r, j] + self.learning_rate*(d[j] - self.weights[min_r, j])

            self.learning_rate = round(self.learning_rate - 0.05, 2)

    def test(self, test_data):
        R = np.zeros(self.output_size)
        res = [[0, []] for i in range(self.output_size)]
        for i, d in enumerate(test_data):
            for j in range(len(R)):
                R[j] = np.linalg.norm(d - self.weights[j])

            min_r = np.argmin(R)
            res[min_r][0] += 1
            res[min_r][1].append(i)
        return res


# Исходные данные о студентах
students_data = np.array([
    ["M", "Y", 60, 79, 60, 72, 63, 1.00],
    ["M", "N", 60, 61, 30, 5, 17, 0.00],
    ["F", "N", 60, 61, 30, 66, 58, 0.00],
    ["M", "Y", 85, 78, 72, 70, 85, 1.25],
    ["F", "Y", 65, 78, 60, 67, 65, 1.00],
    ["F", "Y", 60, 78, 77, 81, 60, 1.25],
    ["F", "Y", 55, 79, 56, 69, 72, 0.00],
    ["M", "N", 55, 56, 50, 56, 60, 0.00],
    ["M", "N", 55, 60, 21, 64, 50, 0.00],
    ["M", "N", 60, 56, 30, 16, 17, 0.00],
    ["F", "Y", 85, 89, 85, 92, 85, 1.75],
    ["F", "Y", 60, 88, 76, 66, 60, 1.25],
    ["M", "N", 55, 64, 0, 9, 50, 0.00],
    ["F", "Y", 80, 83, 62, 72, 72, 1.25],
    ["M", "N", 55, 10, 3, 8, 50, 0.00],
    ["F", "Y", 60, 67, 57, 64, 50, 0.00],
    ["M", "Y", 75, 98, 86, 82, 85, 1.50],
    ["F", "Y", 85, 85, 81, 85, 72, 1.25],
    ["M", "Y", 80, 56, 50, 69, 50, 0.00],
    ["M", "N", 55, 60, 30, 8, 60, 0.00]
])
students_f = np.array([
    "Vardanyan",
    "Gorbunov",
    "Humenyuk",
    "Egorov",
    "Zakharova",
    "Ivanova",
    "Ishonina",
    "Klimchuk",
    "Lisovskiy",
    "Netreba",
    "Ostapova",
    "Pashkova",
    "Popov",
    "Sazon",
    "Steponenko",
    "Terentieva",
    "Titov",
    "Chernova",
    "Chetkin",
    "Shevchenko"
])

train_data = normal(students_data)

knn = Kohonen_Neural_Network(7, 4)
knn.train(train_data)

clusters = knn.test(train_data)
for c in clusters:
    students = c[1]
    sex = ""
    scholarships = ""
    middle_score = 0
    middle_scholarships = 0
    fio = []
    if(len(students) != 0):
        for s in students:
            data = students_data[s]
            sex += data[0]
            scholarships += data[1]
            middle_score += np.sum(list(map(int, data[2:7])))
            middle_scholarships += float(data[7])
            fio.append(students_f[s])
        # print(f'Number of objects in the cluster: {c[0]}, '
        #       f'Sex composition: {list(set(sex))}, '
        #       f'Scholarships composition: {list(set(scholarships))},'
        #       f'Middle score: {round(middle_score/(len(students)*6), 2)}, '
        #       f'Middle scholarships coef: {middle_scholarships/len(students)}')
        print(f'{c[0]}, \t'
              f'{list(set(sex))}, \t'
              f'{list(set(scholarships))}, \t'
              f'{round(middle_score / (len(students) * 6), 2)}, \t'
              f'{round(middle_scholarships / len(students), 2)}, \t'
              f'Familynames: {fio}')