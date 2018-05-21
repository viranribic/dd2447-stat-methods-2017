import numpy as np
from graph import *

class Train:
    graph = Graph()
    D = graph.degree
    V = graph.V
    T = 6                  # Number of observations, the given data
    O = list()              # Observations
    O_true = list()         # True observations, without noise
    path = list()           # Path corresponding to the true observations
    p = 0.05                # Probability associated with noise

    def __init__(self):
        self.O, self.O_true, self.path = self.generateObservationsPath()
        if len(self.O) < self.T:
            self.T = len(self.O)

    def generateObservationsPath(self):
        observations = list()
        true_observations = list()
        path = list()

        # First random observation
        o = np.random.randint(0, self.D)
        observations.append(o)

        # First random vertex and edge feasible with first observation
        current_v = np.random.randint(0, self.V)
        previous_v = current_v
        for i in range(self.V):
            if i != current_v and self.graph.G[current_v, i] == o:
                path.append((current_v, i))
                current_v = i
                break

        # Generate path
        for i in range(1, self.T):
            if o == 0:
                # Then the train can exit either through L or through R
                o = np.random.randint(1, self.D)
                for j in range(self.V):
                    # Make sure not going back and find the edge
                    if j != previous_v and j != current_v and self.graph.G[current_v, j] == o:
                        observations.append(o)
                        path.append((current_v, j))
                        previous_v = current_v
                        current_v = j
                        break
            elif o == 1 or o ==2:
                # Then the train has to exit through 0
                o = 0
                for j in range(self.V):
                    if j != previous_v and j != current_v and self.graph.G[current_v, j] == o:
                        observations.append(o)
                        path.append((current_v, j))
                        previous_v = current_v
                        current_v = j
                        break
        true_observations = observations.copy()
        # Add some noise
        for i in range(self.T):
            r = np.random.randint(1, 101)
            if r < self.p * 100:
                observations[i] = np.random.randint(0, self.D)
        return observations, true_observations, path

    # Probability of going from some position s' to s=(v,e) (i.e., the train
    # has passed v exiting through e) in t steps and observing observations O
    # e = tuple, e[0] = v
    # f = (u,v)
    # g = (w,v)
    def c(self, s, t):
        v = s[0]
        e = s[1]
        u = v
        w = v
        label_e = 3
        label_f = 3
        label_g = 3

        if t == 0:
            return 1./float(self.V)

        else:
            # Look for the label of e in graph.G
            # Look for the switch of v in graph.alphas
            # Look for the neighbors of v and for the connecting edges
            switch = self.graph.alphas[v]
            for i in range(self.V):
                for j in range(self.V):
                    if i==v and j==e[1] and label_e==3: # e found
                        label_e = self.graph.G[i,j]
                    elif i==v and j!=e[1] and self.graph.A[i,j] == 1 and u == v: # first neighbor found
                        u = j
                        f = (u,v)
                        label_f = self.graph.G[v,u]
                    elif i==v and j!=e[1] and self.graph.A[i,j] == 1 and u!=v and w==v: # second neighbor found
                        w = j
                        g = (w,v)
                        label_g = self.graph.G[v,w]

            if label_e==0 and self.O[t]==0:
                return (self.c((u,f), t-1) + self.c((w,g), t-1))*(1.-self.p)
            if label_e==0 and self.O[t]!=0:
                return (self.c((u,f), t-1) + self.c((w,g), t-1))*self.p

            if label_e==1 and switch==1 and self.O[t]==1 and (label_f==0 or label_g==0):
                if label_f==0:
                    return (self.c((u,f), t-1))*(1.-self.p)
                else:
                    return (self.c((w,g), t-1))*(1.-self.p)

            if label_e==1 and switch==1 and self.O[t]!=1 and (label_f==0 or label_g==0):
                if label_f==0:
                    return (self.c((u,f), t-1))*self.p
                else:
                    return (self.c((w,g), t-1))*self.p

            if label_e==2 and switch==2 and self.O[t]==2 and (label_f==0 or label_g==0):
                if label_f==0:
                    return (self.c((u,f), t-1))*(1.-self.p)
                else:
                    return (self.c((w,g), t-1))*(1.-self.p)

            if label_e==2 and switch==2 and self.O[t]!=2 and (label_f==0 or label_g==0):
                if label_f==0:
                    return (self.c((u,f), t-1))*self.p
                else:
                    return (self.c((w,g), t-1))*self.p

            if label_e==1 and switch==2:
                return 0.0
            if label_e==2 and switch==1:
                return 0.0
        # print("No return value for s =", s, "and t = ", t)
        # print("label_e = ", label_e, "label_f = ", label_f, "label_g = ", label_g, "switch = ", switch, "O[t] =", self.O[t])
        return 0.0

    def computeProbObs(self):
        sum_prob = 0.0
        #probabilities = list()

        last = 0
        # for every node, compute the probability of passing that node through
        # each exiting edge
        for v in range(self.V):
            prob = 0.0
            for w in range(self.V):
                if self.graph.G[v, w] != 3 and w != v:
                    # first exiting edge found
                    e = (v, w)
                    last = w
                    break
            prob = self.c((v, e), self.T-1)
            #probabilities.append(prob)
            sum_prob = sum_prob + prob

            for w in range(last + 1, self.V):
                if self.graph.G[v, w] != 3 and w != v:
                    # second exiting edge found
                    e = (v, w)
                    last = w
                    break
            prob = self.c((v, e), self.T-1)
            #probabilities.append(prob)
            sum_prob = sum_prob + prob

            for w in range(last + 1, self.V):
                if self.graph.G[v, w] != 3 and w != v:
                    # third exiting edge found
                    e = (v, w)
                    break
            prob = self.c((v, e), self.T-1)
            #probabilities.append(prob)
            sum_prob = sum_prob + prob

        #highest_prob = max(probabilities)

        return sum_prob
