import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def get_mnist_data():
    data = open("../component_filter_nn/mnist_dataset/mnist_train.csv")

    labels = []
    images = []

    for line in data:
        l = line.strip().split(",")
        l = [int(i) for i in l]
        labels += [l[0]]
        images += [np.array(l[1:])]

    return images,labels

def visualize(image):
    plt.imshow(image.reshape((28,28)))
    plt.show()

