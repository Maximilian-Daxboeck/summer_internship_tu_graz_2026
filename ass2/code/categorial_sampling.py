from random import random
from matplotlib import pyplot as plt

µ = [0.5, 0.1, 0.3, 0.1] #S P(X=0)=0.5, P(X=1)=0.1, ...

def sample(µ, num):
    samples = []
    sum_µ = [sum(µ[:k+1]) for k in range(len(µ))]
    for _ in range(num):
        u = random()

        k = 0
        s = 0
        while u > s:
            s += µ[k]
            k += 1
        samples += [k-1]

    return samples

def make_histogram(samples):
    plt.hist(samples)
    plt.show()

if __name__ == '__main__':
    make_histogram(sample(µ, 1000))


'''samples = []
for _ in range(100):
    samples += sample(µ, 1)
    make_histogram(samples)'''
