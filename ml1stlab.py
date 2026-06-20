import csv
import math

X = []
y = []

with open("diabetes.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)

    for i, row in enumerate(reader, start=2):  
        try:
            X.append([float(v) for v in row[:-1]])
            y.append(int(row[-1]))
        except Exception as e:
            continue


w = [0.0] * len(X[0])
b = 1.0


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def activation(z):
    if sigmoid(z) <= 0.5:
        return 0
    else:
        return 1


def train(epoch, X, w, y, b, eta):

    for _ in range(epoch):

        for row in range(len(X)):

            s = b

            for c in range(len(w)):
                s += X[row][c] * w[c]

            prediction = activation(s)
            error = y[row] - prediction

            if error != 0:
                
                for c in range(len(w)):
                    dldw=2*error*X[row][c]
                    w[c] -= eta * dldw

                b += eta * error

    return w, b


w, b = train(1000, X, w, y, b, 0.1)

print("Optimum weights:", w)
print("Optimum bias:", b)