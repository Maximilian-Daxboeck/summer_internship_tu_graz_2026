import numpy as np
from matplotlib import pyplot as plt
from get_mnist_data import *


def loss(X,Y):
    n = len(X)

    l = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                dX = X[i] - X[j]
                dY = Y[i] - Y[j]
                l += (np.dot(dX,dX)-np.dot(dY,dY))**2

    return l

def grad_y_of_loss(X,Y):
    n = len(X)
    grad = []
    for k in range(n):
        grad += [-8 * sum([(np.dot(X[k] - X[j],X[k] - X[j])-np.dot(Y[k] - Y[j],Y[k] - Y[j])) * (Y[k]-Y[j]) for j in range(n)])]

    return grad


def update_y(X,Y):
    n = len(X)
    grad = grad_y_of_loss(X,Y)
    for k in range(n):
        Y[k] -= 1e-4 * grad[k]

    return Y

def train(X,Y,num_iterations):
    for i in range(num_iterations):
        print(i, loss(X,Y))
        Y = update_y(X,Y)

    return Y

cmap = {
    0:"black",
    1:"red",
    2:"blue",
    3:"green",
    4:"magenta",
    5:"cyan",
    6:"yellow",
    7:"orange",
    8:"purple",
    9:"brown",
}

number_of_data_points = 100

x, labels = get_mnist_data()
x = x[:number_of_data_points]

avg_x = sum(x) / number_of_data_points
avg_x.dtype = np.float32
for i in range(number_of_data_points):
    x[i].dtype = np.float32
    x[i] -= avg_x

labels = labels[:number_of_data_points]
y = [np.random.randn(2) for _ in range(number_of_data_points)]


y_trained = train(x,y, 2000)
for k in range(number_of_data_points):
    l = labels[k]
    plt.scatter(y_trained[k][0], y_trained[k][1], color=cmap[l])
plt.show()
