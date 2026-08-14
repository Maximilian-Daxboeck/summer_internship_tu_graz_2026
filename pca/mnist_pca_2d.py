from get_mnist_data import *
import matplotlib.patches as mpatches

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
#images = images[:10000]

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

pc1 = eigenvectors[:,i]
pc2 = eigenvectors[:,j]

pc1 = pc1.real/np.linalg.norm(pc1.real)
pc2 = pc2.real/np.linalg.norm(pc2.real)

pc1_values = np.dot(pc1, X.transpose(1,0))
pc2_values = np.dot(pc2, X.transpose(1,0))

visualize(pc1)
visualize(pc2)

for i in range(10000):
    l = labels[i]
    plt.scatter(pc1_values[i], pc2_values[i], color = cmap[l], alpha = 0.2)

legend_handles = [
    mpatches.Patch(color=farbe, label=str(ziffer))
    for ziffer, farbe in cmap.items()
]
plt.legend(handles=legend_handles, title="Ziffer")


plt.show()