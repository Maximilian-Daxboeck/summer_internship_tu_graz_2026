import numpy as np



# Sehr einfaches Modell:


features = np.array([[1,0,0],
                     [0,1,0],
                     [0,0,1],
                     [0,0,0],
                     [1,1,1],
                     [1,1,0]])

labels = np.array([[0,1,0,0],
                   [0,1,0,0],
                   [0,1,0,0],
                   [1,0,0,0],
                   [0,0,0,1],
                   [0,0,1,0]])

def prior(class_index):
    return sum(labels.transpose(1,0)[class_index])/6

def likelihood(x_index, class_index):
    n_y = 0
    n_x = 0
    for k in range(6):
        if labels[k][class_index] == 1:
            n_y += 1
            if features[k][x_index] == 1:
                n_x += 1

    return n_x/n_y

def likelihood_x(x, class_index):
    l = 1
    for k in range(3):
        l *= likelihood(k, class_index)

    return l


posterior_matrix = np.array([[likelihood_x(i,j) for i in range(3)] for j in range(4)])

print(posterior_matrix)
