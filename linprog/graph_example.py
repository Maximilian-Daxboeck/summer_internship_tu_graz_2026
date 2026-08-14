import numpy as np
from scipy.optimize import linprog

import networkx as nx
import matplotlib.pyplot as plt

A = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # Max capacity of S->A,B and A,B,C->Z
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],

              [0,0,0,0,0,1,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # Max capacity of A,B,C->A,B,C
              [0,0,0,0,0,-1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,-1,0,0,0,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,-1,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,1,0,0,0,0,0],

              [1,0,-1,0,0,-1,1,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0], # Input = Output
              [0,1,0,-1,0,1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,-1,0,0,1,-1,1,-1,0,0,0,0,0,0,0,0,0,0,0]])

b = np.array([5,10,6,1,4,5,5,1,1,3,3,0,0,0])

c = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # x_SA + x_SB should be maximized

x = linprog(-c, A_eq=A, b_eq=b)

x_values = x.x


x_SA = x_values[0]
x_SB = x_values[1]

x_AB = x_values[5] - x_values[6]
x_AC = x_values[9] - x_values[10]
x_BC = x_values[7] - x_values[8]

x_AZ = x_values[2]
x_BZ = x_values[3]
x_CZ = x_values[4]

print("RESULT: max c^Tx = {}\n\nx_SA = {}\nx_SB = {}\nx_AB = {}\nx_AC = {}\nx_BC = {}\nx_AZ = {}\nx_BZ = {}\nx_CZ = {}\n".format(
    x_SA + x_SB, x_SA, x_SB, x_AB, x_AC, x_BC, x_AZ, x_BZ, x_CZ
))

pos = {"S":(0,.5),
       "A":(.33,.8),
       "B":(.33,.2),
       "C":(.67,.9),
       "Z":(1,.5)
       }

G = nx.DiGraph()

G.add_edge("S","A",weight=x_SA/2)
G.add_edge("S","B",weight=x_SB/2)
if x_AB >= 0:
    G.add_edge("A","B",weight=x_AB/2)
else:
    G.add_edge("B","A",weight=-x_AB/2)
if x_AC >= 0:
    G.add_edge("A","C",weight=x_AC/2)
else:
    G.add_edge("C","A",weight=-x_AC/2)
if x_BC >= 0:
    G.add_edge("B","C",weight=x_BC/2)
else:
    G.add_edge("C","B",weight=-x_BC/2)
G.add_edge("A","Z",weight=x_AZ/2)
G.add_edge("B","Z",weight=x_BZ/2)
G.add_edge("C","Z",weight=x_CZ/2)

edges = G.edges()
widths = [G[u][v]['weight'] for u, v in edges]

nx.circular_layout(G)
nx.draw(G, with_labels=True, node_color='lightgray', node_size=800, edgelist=edges, width=widths, pos=pos)

plt.show()