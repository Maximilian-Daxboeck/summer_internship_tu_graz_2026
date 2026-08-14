import numpy as np
from scipy.optimize import linprog

def get_weird_z_array(m,t,i):
    z = np.zeros(m * t)
    for j in range(m):
        z[(j-1)*m +i] = 1

    return z


def get_optimal_schedule(times, work):
    m = times.shape[1] # m...number of machines -> m - 1 equality constraints because of equal work time of all machines
    t = times.shape[0] # t...number of tasks

    c = np.reshape(times, (m * t,))
    #print(c)
    b = np.append(work, [0 for _ in range(m - 1)])
    #print(b)
    A = np.zeros((t+m-1, m*t))

    for i in range(t):
        A[i, i*m:(i+1)*m]  = np.ones(m)
    #print(A)

    #z = np.zeros(m * t)
    #for i in range(m):
    #    z[i*m] = 1

    z = get_weird_z_array(m,t,0)

    #print(z)

    for i in range(m-1):
        A[t + i, :] = z - get_weird_z_array(m,t,i+1)
        A[t + i, :] = A[t + i, :] * c

    #print(A)



    result = linprog(c, A_eq=A, b_eq=b)

    minimum_time = result.fun / m
    work_distribution = np.reshape(result.x, (t, m))

    return minimum_time, work_distribution


if __name__ == "__main__":
        #    Machines  A, B, C
    times = np.array([[2, 3, 1],  # Time for task 1 in h
                      [2, 1, 5]])  # Time for task 2 in h
    work = np.array([5, 2]) # how often task  1 and 2 have to be done

    minimum_time, work_distribution = get_optimal_schedule(times, work)

    print(minimum_time)
    print(work_distribution)



