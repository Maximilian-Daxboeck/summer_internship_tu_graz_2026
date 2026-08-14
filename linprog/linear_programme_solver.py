from itertools import combinations
import numpy as np


def is_feasible(x,A,b):
    if (x >= 0).all() and np.allclose(A @ x, b):
        return True
    else:
        return False

def is_basic(x,A):
    B = []
    for i in range(len(x)): # get the indices of all dimensions where x!=0
        if x[i] != 0:
            B += [i]
    if len(B) == A.shape[0]: # the number of non-zero dimensions of x (i.e. the number of basic variables) must equal the number of equations
        if det(compute_A_B(A,B)) != 0:
            return True
        else:
            return False
    else:
        return False

def compute_A_B(A,B): # A...matrix of equations, B...index list of basis -> returns a matrix which only contains the columns with index in B
    A = A.transpose(1,0)
    A_B = []
    for i in B:
        A_B += [A[i]]
    A_B = np.array(A_B).transpose(1,0)

    return A_B


def det(A):
    return np.linalg.det(A)

def solve_system(A,b): # solves Ax=b for x, provided that det(A)!=0
    return np.linalg.solve(A,b)

def inv(A):
    if det(A) != 0:
        return np.linalg.inv(A)
    else:
        return "ERROR: Singular matrix!"


def get_first_basic_solution_by_trying(A,b): # returns a basic feasible solution to start the simplex algorithm (by trying)
    number_of_equations = A.shape[0]
    number_of_variables = A.shape[1]
    variable_indices = list(range(number_of_variables))
    b_combinations = [list(c) for c in combinations(variable_indices, number_of_equations)]

    for B in b_combinations:
        A_B = compute_A_B(A,B)
        if det(A_B)==0:
            continue
        else:
            x_ = solve_system(A_B,b)
            x = np.zeros(number_of_variables)
            for i in range(number_of_equations): # creating the basic solution from the solution of the linear system of equations and setting the rest to zero => x is already basic
                x[B[i]] = x_[i]

            if is_feasible(x,A,b): # => yes: x is a basic feasible solution -> return x, otherwise try with new B
                return x, B
            else:
                continue

    # if there exists no possible feasible solution to start with, the problem is infeasible
    return "ERROR: The program does not have any basic feasible solutions" # <--`

def optimize_by_trying(A,b,c):
    number_of_equations = A.shape[0]
    number_of_variables = A.shape[1]
    variable_indices = list(range(number_of_variables))
    b_combinations = [list(c) for c in combinations(variable_indices, number_of_equations)]

    X = []

    for B in b_combinations:
        A_B = compute_A_B(A, B)
        if det(A_B) == 0:
            continue
        else:
            x_ = solve_system(A_B, b)
            x = np.zeros(number_of_variables)
            for i in range(
                    number_of_equations):  # creating the basic solution from the solution of the linear system of equations and setting the rest to zero => x is already basic
                x[B[i]] = x_[i]

            if is_feasible(x, A, b):  # => yes: x is a basic feasible solution -> return x, otherwise try with new B
                X += [x]
            else:
                continue

    X.sort(key=lambda x: np.dot(c,x))

    return X[-1], np.dot(c,X[-1])


class SimplexTable:
    def __init__(self, A, b, c, x0, B0): # assuming that the rows of A are linearly independent and x0 is a basic feasible solution with basis B0
        self.number_of_vars = A.shape[1]
        self.number_of_equations = A.shape[0]
        self.basic_vars = []
        self.non_basic_vars = []
        for i in range(self.number_of_vars): # sorts the indices of the variables according to whether they are basic or not
            if i in B0:
                self.basic_vars += [i]
            else:
                self.non_basic_vars += [i]

        A_B0 = compute_A_B(A,B0) # matrix with the columns corresponding to the basis variables

        inv_A_B0 = inv(A_B0) # inverse always exists because B0 is a basis
        self.array = inv_A_B0 @ A
        self.values = inv_A_B0 @ b # =x0, values of the initial basic variables

        self.array = np.delete(self.array, B0, axis=1) # deletes the columns in the array which correspond to the basic variables (because these are brought to the other side)
        self.array = -self.array # because when rearranging we have to subtract the coefficients of all basic variables

        # basic_vars, non_basic_vars, array, values : DONE

        self.z = 0                                                    # initializing z and array_for_z
        self.array_for_z = np.zeros(len(self.non_basic_vars))
        for i in range(len(self.non_basic_vars)):
            self.array_for_z[i] = c[self.non_basic_vars[i]]           # encodes the influence of the non-basic variables on their coefficients according to c

        for i in range(len(self.basic_vars)):                         # if a basic variable is in the optimization objective function this code performs a substitution
            factor = c[self.basic_vars[i]]
            self.z += factor * self.values[i]
            self.array_for_z += factor * self.array[i]

        # EVERYTHING DONE! The Simplex algorithm can get started now...


    def read_solution_from_table(self): # reads the current solution for x off the table and returns the full vector x and the target value z
        x = np.zeros(self.number_of_vars)
        for i in range(self.number_of_vars):
            if i in self.basic_vars:
                j = self.basic_vars.index(i)
                x[i] = self.values[j]
            else:
                pass # (non-basic-variables are always equal to zero)

        return x, self.z # returns optimal values for x and the optimal target value z


    def pivot_step(self):
        if (self.array_for_z > 0).any():
            index_of_entering_var_in_non_basic_vars = np.argmax(self.array_for_z)
            entering_var = self.non_basic_vars[index_of_entering_var_in_non_basic_vars] # index ot the variable which enters the basis

            scores = [] # by how much at maximum can I increase each the entering variable such that each basic variable is not negative
            for i in range(len(self.basic_vars)):
                if self.array[i][index_of_entering_var_in_non_basic_vars] < 0:
                    scores += [-self.values[i]/self.array[i][index_of_entering_var_in_non_basic_vars]] # y = b + ax, since y>=0 -> -b/a >= x, the '<='-sign is changed because a is negative, a...coefficient of x, b...value of y, y...basic var, x...entering non-basic var
                else:
                    scores += ["NO BOUNDS"]


            if (np.array(scores) == "NO BOUNDS").all():
                return "ERROR: unbounded"

            else:
                for i in range(len(scores)):
                    if scores[i] == "NO BOUNDS":
                        scores[i] = 1e100

                index_of_exiting_var_in_basic_vars = np.argmin(scores) # the index of the maximum amount at which the entering var can be changed such that all basic vars remain non-negative
                exiting_var = self.basic_vars[index_of_exiting_var_in_basic_vars]


                self.basic_vars[index_of_exiting_var_in_basic_vars] = entering_var # swapping the names of the exiting and entering var in the tableau
                self.non_basic_vars[index_of_entering_var_in_non_basic_vars] = exiting_var

                factor = self.array[index_of_exiting_var_in_basic_vars][index_of_entering_var_in_non_basic_vars] # update changed line (the line of the exiting variable) This is just a step-by-step formalization of rearranging
                self.array[index_of_exiting_var_in_basic_vars][index_of_entering_var_in_non_basic_vars] = -1
                self.array[index_of_exiting_var_in_basic_vars] /= -factor
                self.values[index_of_exiting_var_in_basic_vars] /= -factor


                for i in range(self.number_of_equations): # substituting the entered basis variable by its line in self.array
                    if i != index_of_exiting_var_in_basic_vars:
                        factor = self.array[i][index_of_entering_var_in_non_basic_vars]
                        self.array[i][index_of_entering_var_in_non_basic_vars] = 0
                        self.array[i] += factor * self.array[index_of_exiting_var_in_basic_vars]
                        self.values[i] += factor * self.values[index_of_exiting_var_in_basic_vars]


                factor = self.array_for_z[index_of_entering_var_in_non_basic_vars] # the same for array_for_z and z
                self.array_for_z[index_of_entering_var_in_non_basic_vars] = 0
                self.array_for_z += factor * self.array[index_of_exiting_var_in_basic_vars]
                self.z += factor * self.values[index_of_exiting_var_in_basic_vars]


                return "STEP SUCCESSFUL"

        else:
            return "DONE: Maximum already reached"

    def optimize(self):
        done = False
        while not done:
            result = self.pivot_step()
            if result == "DONE: Maximum already reached":
                return self.read_solution_from_table()
            elif result == "ERROR: unbounded":
                return "ERROR: unbounded"
            else:
                continue


def get_first_feasible_basic_solution(A, b):
    number_of_equations = A.shape[0]

    id_matrix = np.identity(number_of_equations)
    A_ = np.hstack((A, id_matrix))                            # system of linear equations with additional variables (which should go to zero in the optimization process) and with trivial basic feasible solution (-> Simplex can get started right away)

    B0 = [A.shape[1]+i for i in range(number_of_equations)]    # basis contains the new vars

    # Irrelevant code:
    # A_B0 = compute_A_B(A_,B0) -> identity matrix
    # x0 = solve_system(A_B0,b) -> x=b

    x0 = b

    c = np.zeros(A_.shape[1])                                 # creates the c which minimizes the additional variables (or maximizes their negative counterpiece)
    for i in B0:
        c[i] = -1

    simplex_table = SimplexTable(A_,b,c,x0,B0)

    x , z = simplex_table.optimize() # gets x and z of the step 1 problem
    B = simplex_table.basic_vars
    B.sort()

    if z == 0:
        return x[:A.shape[1]] , B # x must be cut off because it also contains zeros for the additional variables which do not exist in the original LP
    else:
        return "ERROR: The LP is infeasible"

def solve_LP(A,b,c):                            # returns (Message, x, z)
    result = get_first_feasible_basic_solution(A,b)
    if result == "ERROR: The LP is infeasible":
        return result, None, None
    else:
        x0 , B0 = result

        st = SimplexTable(A,b,c,x0,B0)
        result2 = st.optimize()
        if result2 == "ERROR: unbounded":
            return result2, None, None
        else:
            x, z = result2
            return "SUCCESS: found a solution", x, z






def main():
    '''
    ------------------- TEST CODE ------------------------
    A = np.array([[1,3,1,2],
                  [0,2,1,-1]])
    b = np.array([4,2])

    c = np.array([1,2,0,-1])

    # x0 , B0 = get_first_basic_solution_by_trying(A,b)

    x0 = np.array([2,0,2,0])
    B0 = [0,2]

    st = SimplexTable(A,b,c,x0,B0)

    x_opt, z_opt = st.optimize()

    print(x_opt)
    print(z_opt)

    -----------------------------------------------

    A = np.array([[1,2,-1,0],
                  [2,-1,0,1]])
    b = np.array([4,5])
    c = np.array([1,1,0,0])

    x0, B0 = get_first_feasible_basic_solution(A,b)

    st = SimplexTable(A,b,c,x0,B0)
    print(st.optimize())
    ----------------------------------------------
    '''

    A = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Max capacity of S->A,B and A,B,C->Z
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

         [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Max capacity of A,B,C->A,B,C
         [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

         [1, 0, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Input = Output
         [0, 1, 0, -1, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    b = np.array([5, 10, 6, 1, 4, 5, 5, 1, 1, 3, 3, 0, 0, 0])

    c = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    msg, x, z = solve_LP(A,b,c)
    print(msg, x, z)
    import scipy
    print(scipy.optimize.linprog(-c, A_eq=A, b_eq=b).x)



if __name__ == "__main__":
    main()





