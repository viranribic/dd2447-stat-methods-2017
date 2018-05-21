import matplotlib.pyplot as plt
from graph import *
from train import *
from mcmc import *
from gibbs import *

train = Train()

print("Tracks:")
print(train.graph.G)

print("Observations:")
print(train.O)

print("True observations:")
print(train.O_true)

print("True path:")
print(train.path)

original_settings = train.graph.alphas


# Metropolis Hastings:

setting_probs, ml_settings, node1 = metropolis_hastings(1000, train)

print("True switch settings:")
print(original_settings)
print("Most likely switch settings according to Metropolis Hastings:")
print(ml_settings)

print("MCMC samples:", len(setting_probs))
plt.plot(setting_probs)
plt.title("Metropolis-Hastings MCMC sampling")
plt.ylabel("Probability")
plt.xlabel("Sample")
plt.show()

print("Node 1 proposals:")
plt.plot(node1)
plt.title("Metropolis-Hastings MCMC sampling: Node 1 proposals")
plt.ylabel("Label")
plt.xlabel("Iteration")
plt.show()

pL = 0
pR = 0
for i in node1:
    if i == 1:
        pL = pL + 1
    else:
        pR = pR + 1
pL = pL/len(node1)
pR = pR/len(node1)
print("For node 1 the probabilities associated to each switch setting are: L = ", pL, ", R = ", pR)



# # Gibbs sampling:
# train.graph.setAlphas(original_settings)
# settings_prob_gibbs, ml_gibbs, node1_gibbs = GibbsSampling(10000, train)
#
# print("True switch settings:")
# print(original_settings)
# print("Most likely switch settings according to Gibbs sampling:")
# print(ml_gibbs)
#
# print("Gibbs samples:", len(settings_prob_gibbs))
# plt.plot(settings_prob_gibbs[(len(settings_prob_gibbs)-1000):len(settings_prob_gibbs)])
# plt.title("Gibbs sampling MCMC sampling")
# plt.ylabel("Probability")
# plt.xlabel("Sample")
# plt.show()
#
# print("Node 1 proposals with Gibbs:")
# plt.plot(node1_gibbs[(len(node1_gibbs)-1000):len(node1_gibbs)])
# plt.title("Node 1 proposals with Gibbs")
# plt.ylabel("Label")
# plt.xlabel("Iteration")
# plt.show()
#
# pL = 0
# pR = 0
# for i in node1_gibbs:
#     if i == 1:
#         pL = pL + 1
#     else:
#         pR = pR + 1
# pL = pL/len(node1_gibbs)
# pR = pR/len(node1_gibbs)
# print("Using Gibbs, for node 1 the probabilities associated to each switch setting are: L = ", pL, ", R = ", pR)




# Blocked Gibbs sampling:
# train.graph.setAlphas(original_settings)
# settings_prob_blocked_gibbs, ml_blocked_gibbs, node1_blocked_gibbs = blockedGibbsSampling(10000, train, 2)
#
# print("True switch settings:")
# print(original_settings)
# print("Most likely switch settings according to Gibbs sampling:")
# print(ml_blocked_gibbs)
#
# print("Gibbs samples:", len(settings_prob_blocked_gibbs))
# plt.plot(settings_prob_blocked_gibbs[(len(settings_prob_blocked_gibbs)-1000):len(settings_prob_blocked_gibbs)])
# plt.title("Blocked Gibbs sampling MCMC sampling")
# plt.ylabel("Probability")
# plt.xlabel("Sample")
# plt.show()
#
# print("Node 1 proposals with blocked Gibbs:")
# plt.plot(node1_blocked_gibbs[(len(node1_blocked_gibbs)-1000):len(node1_blocked_gibbs)])
# plt.title("Node 1 proposals with Blocked Gibbs")
# plt.ylabel("Label")
# plt.xlabel("Iteration")
# plt.show()
#
# pL = 0
# pR = 0
# for i in node1_blocked_gibbs:
#     if i == 1:
#         pL = pL + 1
#     else:
#         pR = pR + 1
# pL = pL/len(node1_blocked_gibbs)
# pR = pR/len(node1_blocked_gibbs)
# print("Using Gibbs, for node 1 the probabilities associated to each switch setting are: L = ", pL, ", R = ", pR)
