import numpy as np

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def softmax_strong(x):
    a = np.max(x)
    return softmax(x-a)

x1 = np.array([1000,999,998])

print(softmax(x1), softmax_strong(x1))