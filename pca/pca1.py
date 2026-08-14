import numpy as np

n = 10 # Number of Samples
d = 3 # Dimensionality of Data

X = np.random.randn(n,d) # -> langes array mit n d-dimensionalen Samples

cov = 1/n * X.transpose(1,0) @ X
eig = np.linalg.eig(cov)
eigenvalues, eigenvectors = eig[0], eig[1]
i = np.argmax(eigenvalues)
eigenvalues[i] = 0
j = np.argmax(eigenvalues)
pc1 = eigenvectors[:, i]
pc2 = eigenvectors[:, j]

X_projected = X- np.outer((X @ pc1) , pc1)

cov_new = 1/n * X_projected.transpose(1,0) @ X_projected
eig = np.linalg.eig(cov_new)
eigenvalues_new, eigenvectors_new = eig[0], eig[1]
k = np.argmax(eigenvalues_new)
pc2_new = eigenvectors_new[:, k]

print(pc2)
print(pc2_new)


