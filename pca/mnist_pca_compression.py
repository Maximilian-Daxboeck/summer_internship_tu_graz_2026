from get_mnist_data import *

images, labels = get_mnist_data()

def get_mapping(data, n):
    X = np.array(data)

    mu = np.mean(X,axis=0)
    X = X - mu
    cov = np.matmul(X.transpose(1,0), X)/60000

    eig = np.linalg.eig(cov)
    eigenvalues = eig[0]
    eigenvectors = eig[1]

    principal_components = []
    for _ in range(n):
        i = np.argmax(eigenvalues)
        eigenvalues[i] = 0
        principal_components += [eigenvectors[:,i].real]

    return np.array(principal_components).transpose((1,0)), np.array([mu]).transpose(1,0)

def encode(x, B, mu):
    B_t = B.transpose()
    pseudo_inv = np.linalg.inv(B_t @ B) @ B_t
    return pseudo_inv @ (x - mu)

def decode(x, B, mu):
    return B @ x + mu

for k in range(2,400):
    mapping , mu = get_mapping(images, k)
    x = np.array([images[0]]).transpose(1,0)
    x_ = decode(encode(x, mapping, mu), mapping, mu)

    #visualize(x)
    visualize(x_)