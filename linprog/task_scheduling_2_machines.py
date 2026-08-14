import numpy as np
from scipy.optimize import linprog


def get_optimal_schedule(times, work):
    c = np.reshape(times, (4,))
    b = np.append(work,0)
    A = np.array([[1,1,0,0],
                  [0,0,1,1],
                  [times[0][0], -times[0][1], times[1][0], -times[1][1]]])

    result = linprog(c,A_eq=A, b_eq=b)

    minimum_time = result.fun / 2
    work_distribution = np.reshape(result.x,(2,2))

    return minimum_time, work_distribution

if __name__ == "__main__":
    #        Machines  A, B
    times = np.array([[1, 2],  # Time for task 1 in h
                      [2, 1]])  # Time for task 2 in h
    work = np.array([5, 2])

    minimum_time, work_distribution = get_optimal_schedule(times, work)

    print("The minimum running time is {}h.".format(minimum_time))
    print("Machine A should do {} times task 1 and {} times task 2.".format(work_distribution[0][0], work_distribution[1][0]))
    print("Machine B should do {} times task 1 and {} times task 2.".format(work_distribution[0][1], work_distribution[1][1]))



