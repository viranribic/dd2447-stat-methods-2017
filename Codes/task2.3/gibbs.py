import numpy as np
import collections

# To perform Gibbs sampling, at each iteration we have to sample
# just one component of the switche settings
# With probability 0.5 alpha(v_i) remains the same,
# With probability 0.5 alpha(v_i) changes
def getSampleSwitchComponent(oldAlphas, i):
    newAlphas = oldAlphas.copy()
    r = np.random.randint(0, 2)  # r can be either 0 or 1
    if r == 1:
        if newAlphas[i] == 2:
            newAlphas[i] = 1
        else:
            newAlphas[i] = 2
    return newAlphas

def GibbsSampling(num_samples, train):
    burn_in = int(num_samples / 2)  # Number of samples to discard
    N = num_samples + burn_in       # Number of samples to be considered

    # Initialization
    alphas = train.graph.alphas
    alphas_p = train.computeProbObs()
    samples = list()              # Sampled switch settings
    probabilities = list()        # Probabilities associated to the sampled switch settings

    for i in range(N):
        # Sample one switch component at a time
        for j in range(train.V):
            alphas = getSampleSwitchComponent(alphas, j)
            train.graph.setAlphas(alphas)
            alphas_p = train.computeProbObs()
            samples.append(alphas)
            probabilities.append(alphas_p)

    # Find the most likely switch settings according to the probabilities observed
    temp = [tuple(lst) for lst in samples]

    node1 = list()
    for i in temp:
        node1.append(i[0])

    counter = collections.Counter(temp)
    most_likely = counter.most_common(1)[0]
    most_likely = list(most_likely[0])

    return probabilities, most_likely, node1

def getNewAlphaBlock(old_alpha):
    # Flip random switch
    n = len(old_alpha)
    new_alpha = old_alpha.copy()
    new_switch = np.random.randint(0, n)
    if new_alpha[new_switch] == 2:
        new_alpha[new_switch] = 1
    else:
        new_alpha[new_switch] = 2
    return new_alpha

def blockedGibbsSampling(num_samples, train, n):
    burn_in = int(num_samples / 2)  # Number of samples to discard
    N = num_samples + burn_in       # Number of samples to be considered

    # Initialization
    alphas = train.graph.alphas
    alphas_p = train.computeProbObs()
    samples = list()              # Sampled switch settings
    probabilities = list()        # Probabilities associated to the sampled switch settings
    r = len(alphas) % n
    m = int(len(alphas)/n)
    if r > 0:
        m = m + 1

    for i in range(N):
        # Sample one switch component at a time
        for j in range(m):
            if r > 0:
                if j < m - 1:
                    alphas[(0+j*n):(n+j*n)] = getNewAlphaBlock(alphas[(0+j*n):(n+j*n)])
                else:
                    alphas[(n-r):n] = getNewAlphaBlock(alphas[(n-r):n])
            else:
                alphas[(0+j*n):(n+j*n)] = getNewAlphaBlock(alphas[(0+j*n):(n+j*n)])
            train.graph.setAlphas(alphas)
            alphas_p = train.computeProbObs()
            samples.append(alphas)
            probabilities.append(alphas_p)

    # Find the most likely switch settings according to the probabilities observed
    temp = [tuple(lst) for lst in samples]

    node1 = list()
    for i in temp:
        node1.append(i[0])

    counter = collections.Counter(temp)
    most_likely = counter.most_common(1)[0]
    most_likely = list(most_likely[0])

    return probabilities, most_likely, node1
