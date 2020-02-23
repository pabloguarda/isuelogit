import sys

import matplotlib.pyplot as plt

import networkx as nx

folder_plots = "TransportAI/plots/"

#Four Grids
G = nx.grid_2d_graph(3, 3)  # 5x5 grid

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)
# write edgelist to grid.edgelist
nx.write_edgelist(G, path= folder_plots + "/grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path= folder_plots + "grid.edgelist", delimiter=":")

nx.draw(H)
plt.show()

#Erdos Reny

n = 10  # 10 nodes
m = 20  # 20 edges

G = nx.gnm_random_graph(n, m)

# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)

nx.draw(G)
plt.show()

