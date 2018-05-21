####################################################################
#         2.5 Stochastic volatility unknown parameter              #
####################################################################
setwd("C:/Users/Vale/Dropbox/Poli/Erasmus/KTH/Statitical Methods for Applied Computer Science/Assignments/Assignment 2/GitHub/task5")
source("task5_functions.R")

library(invgamma)
library(pscl)
library(ggplot2)

# Question 11
#-------------------------------------------------------------------
T = 500
num_samples = 10000
burn_in = num_samples/2
N = burn_in+num_samples

phi = 0.970944

X = NULL        # will have dimension TXN
sigma2 = NULL   # will have dimension 1xN
beta2 = NULL    # will have dimension 1xN

y = read.table("output.txt")

# Initialization of sigma2 and X
sigma = 0.1315^2
x = array(0,T+1)
x[1] =  rnorm(1, 0, sqrt(sigma/(1-phi^2)))
for (t in 2:(T+1))
  x[t] = rnorm(1, phi*x[t-1], sqrt(sigma))

# Gibbs sampling
for (i in 1:N){
  sigma = sqrt(update_sigma2(x, phi, T))
  
  if (i > burn_in)
    sigma2 = c(sigma2, sigma^2)
  
  x = update_x(sigma^2, phi, T)
  beta = sqrt(update_beta2(x, y[,1], T))
  
  if (i > burn_in)
    beta2 = c(beta2, beta^2)
  
}


x11()
qplot(sigma2, geom="histogram", fill=I("blue"), col=I("grey"), main="Marginal distribution for Sigma") 

x11()
plot(sigma2, type="l")

x11()
qplot(beta2, geom="histogram", fill=I("blue"), col=I("grey"), main="Marginal distribution for Beta") 

x11()
plot(beta2, type="l")



# Question 12
#-------------------------------------------------------------------
T = 500
num_samples = 1000
burn_in = num_samples/2
N = burn_in+num_samples
phi = 0.970944

sigma2 = NULL   # will have dimension 1xN
beta2 = NULL    # will have dimension 1xN

y = read.table("output.txt")

# Initialization of sigma2 and X:
# sigma is initialized to its correct value
# X is the matrix representing the results of each step of SMC
sigma = 0.1315
beta = 0.63715^2
X_smc = smc_tot(sigma, phi, sqrt(beta), T)

# j = sample(1:N, 1)
# x = X_smc[,j]

x = array(0,T+1)
x[1] =  rnorm(1, 0, sqrt(sigma^2/(1-phi^2)))
for (t in 2:(T+1))
  x[t] = rnorm(1, phi*x[t-1], sigma)

# Gibbs sampling
for (i in 1:N){
  sigma = update_sigma2(x, phi, T)
  
  if (!is.na(sigma) & i>burn_in)
    sigma2 = c(sigma2, sigma)
  
  if (is.na(sigma))
    sigma = 0.1315^2

  j = sample(1:999, 1)
  x = smc(X_smc, sqrt(sigma), phi, sqrt(beta), T, j)
  
  beta = update_beta2(x, y[,1], T)
  
  if (i>burn_in)
    beta2 = c(beta2, beta)
}

x11()
qplot(sigma2, geom="histogram", fill=I("blue"), col=I("grey"), main="Marginal distribution for Sigma") 

x11()
plot(sigma2, type="l")

x11()
qplot(beta2, geom="histogram", fill=I("blue"), col=I("grey"), main="Marginal distribution for Beta") 

x11()
plot(beta2, type="l")

