# This script is used to create a plausible graph
# G_ij = {0,L,R} indicates that there exists an edge leaving node i
# and ending in node j. The label of this edge is G_ij.
# So, the train could pass vertex i through edge (i,j) labelled G_ij

# Assumtions:
# label 0 : 0
# label 1 : L
# label 2 : R
# label 3: the edge doesn't exist

import random
import numpy as np

class Graph:
    # Attributes of the graph
    V = 6
    degree = 3
    A = np.zeros(shape=(V,V)) # Adjacency matrix
    G = np.empty(shape=(V,V)) # Matrix representing the actual graph
    G.fill(3)
    alphas = np.empty(shape=V)

    def __init__(self):
        self.createGraph()
        self.generateNewAlphas()

    def countPrevNeighbors(self,j):
        count = 0
        for i in range(0,j):
            if self.A[i,j] == 1:
                count = count + 1
        return count

    # Select 3 random neighbors, in such a way that we obtain a plausible network
    # (i.e. A must be symmetric)
    def setNeighbors(self):
        self.A = np.zeros(shape=(self.V,self.V))
        for i in range(self.V-1):
            diff = self.degree - self.countPrevNeighbors(i)
            if diff > 0:
                neighbors = random.sample(range(i+1, self.V), diff)
                for n in neighbors:
                    self.A[i,n] = 1
                    self.A[n,i] = 1

    # Given a vertex i, we label the neighbors assigning the 3 labels
    def labelNeighbors(self, i):
        orderLabels = random.sample(range(0, self.degree), 3)
        count = 0
        for j in range(self.V):
            if count == 3:
                break
            if self.A[i,j] == 1:
                self.G[i,j] = orderLabels[count]
                count = count + 1

    def createGraph(self):
        self.setNeighbors()
        for i in range(self.V):
            self.labelNeighbors(i)

    # For each node of the graph, randomly generate the switch setting,
    # i.e., set the switch equal to 1 (L) or 2 (R) in case the train enters the
    # vertex from 0
    def generateNewAlphas(self):
        self.alphas = np.random.randint(1, 2 + 1, size = self.V)

    def setAlphas(self, newAlpha):
        self.alphas = newAlpha
