from get_mnist_data import *
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

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

images, labels = get_mnist_data()

X = np.array(images)

mu = np.mean(X,axis=0)
X = X - mu
cov = np.matmul(X.transpose(1,0), X)/60000

eig = np.linalg.eig(cov)
eigenvalues = eig[0]
eigenvectors = eig[1]

i = np.argmax(eigenvalues)
eigenvalues[i] = 0
j = np.argmax(eigenvalues)
eigenvalues[j] = 0
k = np.argmax(eigenvalues)

pc1 = eigenvectors[:,i]
pc2 = eigenvectors[:,j]
pc3 = eigenvectors[:,k]

pc1 = pc1.real/np.linalg.norm(pc1.real)
pc2 = pc2.real/np.linalg.norm(pc2.real)
pc3 = pc3.real/np.linalg.norm(pc3.real)

pc1_values = np.dot(pc1, X.transpose(1,0))
pc2_values = np.dot(pc2, X.transpose(1,0))
pc3_values = np.dot(pc3, X.transpose(1,0))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(1000):
    l = labels[i]
    ax.scatter(pc1_values[i], pc2_values[i], pc3_values[i], color = cmap[l], marker = 'o', alpha=0.8)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

legend_handles = [
    mpatches.Patch(color=farbe, label=str(ziffer))
    for ziffer, farbe in cmap.items()
]
plt.legend(handles=legend_handles, title="Ziffer")


plt.show()
