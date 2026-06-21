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
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1 + ez)




def train(epoch, X, w, y, b, eta):

    for _ in range(epoch):

        for row in range(len(X)):

            s = b

            for c in range(len(w)):
                s += X[row][c] * w[c]

            prediction = sigmoid(s)
            error = prediction - y[row]

            for c in range(len(w)):
                w[c] -= eta * error * X[row][c]

            b -= eta * error

    return w, b
def predict(x):
    s = b

    for c in range(len(w)):
        s += x[c] * w[c]
    if sigmoid(s)>0.5:
        print("Diabetes Predection:Diabetes ")
    else :
        print("Diabetes predection:safe")

    


w, b = train(100, X, w, y, b, 0.01)

print("Optimum weights:", w)
print("Optimum bias:", b)

predict(X[0]);

