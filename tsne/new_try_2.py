import numpy as np
from matplotlib import pyplot as plt
#from get_mnist_data import *

def euclidean_distance_sq(v1, v2): # v1, v2 -> ||v2 - v1||^2
    dv = v2 - v1
    return np.dot(dv, dv)

def p(i,j,X,sigma_sq): # returns p_{i,j} (= similarity between X[i] and X[j]) for given data X
    if i==j:
        return 0
    else:
        n = len(X)
        sum_i = sum([np.exp(-euclidean_distance_sq(X[i], X[k]) / (2 * sigma_sq[i])) if k!=i else 0
                     for k in range(n)])
        sum_j = sum([np.exp(-euclidean_distance_sq(X[j], X[k]) / (2 * sigma_sq[j])) if k!=j else 0
                     for k in range(n)])
        dx_sq = euclidean_distance_sq(X[i], X[j])
        p_ji = np.exp(-dx_sq / (2 * sigma_sq[i]))/sum_i
        p_ij = np.exp(-dx_sq / (2 * sigma_sq[j]))/sum_j

        return (p_ij+p_ji) / (2*n)

def u(i,j,Y): # returns 1/(1 + ||Y[i]-Y[j]||^2) for the formula for q_{i,j}
    return 1/(1 + euclidean_distance_sq(Y[i], Y[j]))

def q(i,j,Y):
    if i==j:
        return 0
    else:
        n = len(Y)
        U = np.array([[u(i,j,Y) for j in range(n)] for i in range(n)])

        sum_ = sum([ sum([ U[i,j]
            if l!=k else 0
            for l in range(n)])
            for k in range(n)])

        return U[i,j] / sum_

def D_KL(X,Y,sigma_sq):
    n = len(X)
    high_dim_sim = np.array([[p(i,j,X,sigma_sq) for i in range(n)] for j in range(n)])
    low_dim_sim = np.array([[q(i,j,Y) for i in range(n)] for j in range(n)])

    return sum([sum([ high_dim_sim[i,j] * np.log(high_dim_sim[i,j] / low_dim_sim[i,j])
                if i!=j else 0
                for i in range(n)])
                for j in range(n)])

def grad_D_KL(X,Y,sigma_sq):
    n = len(X)
    return np.array([4 * sum([(p(i,j,X,sigma_sq)-q(i,j,Y)) * u(i,j,Y) * (Y[i]-Y[j]) for j in range(n)])
                    for i in range(n)])

def train(X,Y_0,sigma_sq,num_iterations, alpha, beta):
    Y_old = Y_0.copy()
    Y = Y_0.copy()
    for iteration in range(num_iterations):
        grad = grad_D_KL(X,Y,sigma_sq)

        Y_new = Y - alpha*grad + beta*(Y-Y_old)

        Y_old = Y
        Y = Y_new

        print(iteration + 1, D_KL(X,Y,sigma_sq))

        for i in range(20):
            plt.scatter(Y[i][0], Y[i][1], color=labels[i], alpha=0.5)

        plt.show()

    return Y




if __name__ == "__main__":
    '''
    X = [
        np.array([1.,1.,4.,3.]),
        np.array([1.,1.,4.,4.]),
        np.array([-1.,-2.,0.,-3.])
    ]
    Y_0 = np.array([np.random.randn(2) for _ in range(3)])
    labels = ["red", "red", "blue"]
    sigma_sq = np.ones(3)

    Y = train(X,Y_0,sigma_sq, 500, 0.01, 0.7)
    '''

    X = [np.random.multivariate_normal(mean=np.array([10.,9.,8.,7.,6.,5.]), cov=np.identity(6)) for _ in range(10)] + [np.random.multivariate_normal(mean=np.array([-1.,-2.,-3.,0.,4.,5.]), cov=np.identity(6)) for _ in range(10)]
    Y_0 = np.array([np.random.randn(2) for _ in range(20)])
    labels = ["red" for _ in range(10)] + ["blue" for _ in range(10)]
    sigma_sq = np.ones(20)

    Y = train(X,Y_0,sigma_sq, 500, 1, 0.9)

    for i in range(20):
        plt.scatter(Y[i][0], Y[i][1], color=labels[i], alpha=0.5)

    plt.show()






