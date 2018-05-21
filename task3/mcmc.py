import numpy as np
import collections

# Generate randomly a new switch settings vector by changing just one switch setting
def getNewAlpha(old_alpha, V):
    # Flip random switch
    new_alpha = old_alpha.copy()
    new_switch = np.random.randint(0, V)
    if new_alpha[new_switch] == 2:
        new_alpha[new_switch] = 1
    else:
        new_alpha[new_switch] = 2
    return new_alpha

# Metropolis Hastings algorithm
# This function returns a list of probabilities, the ones associated to the samples
def metropolis_hastings(num_samples, train):
    burn_in = int(num_samples / 2)  # Number of samples to discard
    N = num_samples + burn_in       # Number of samples to be considered

    # Initialization
    oldAlphas = train.graph.alphas
    oldAlphas_p = train.computeProbObs()
    samples = list()                # Sampled switch settings
    probabilities = list()          # Probabilities associated to the samples

    for i in range(N):
        # Generate a new random switch
        newAlphas = getNewAlpha(oldAlphas, train.V)

        # Acceptance probability
        oldAlphas_p = train.computeProbObs()
        train.graph.setAlphas(newAlphas)
        newAlphas_p = train.computeProbObs()
        acceptance = newAlphas_p / oldAlphas_p # Might give 0-division
        r = min(acceptance, 1)

        if i > burn_in:
            probabilities.append(newAlphas_p) # Save the generated probability if we have passed the burn in part

            u = np.random.randint(1, 101)
            if u < r * 100:
                # accept proposal and set the swtich settings
                oldAlphas = newAlphas
                oldAlphas_p = newAlphas_p
                samples.append(oldAlphas)
                probabilities[-1] = oldAlphas_p # Change to the better probability
            else :
                # keep the old switch settings
                train.graph.setAlphas(oldAlphas)

    # Find the most likely switch settings according to the probabilities observed
    temp = [tuple(lst) for lst in samples]

    node1 = list()
    for i in temp:
        node1.append(i[0])

    counter = collections.Counter(temp)
    most_likely = counter.most_common(1)[0]
    most_likely = list(most_likely[0])

    return probabilities, most_likely, node1
