####################################################################
#         2.1 SMC for the stochastic volatility model              #
####################################################################
setwd("C:/Users/Vale/Dropbox/Poli/Erasmus/KTH/Statitical Methods for Applied Computer Science/DD2447_2017/task1")
source("task1_functions.R")

# Question 1
#-------------------------------------------------------------------

y = read.table("output.txt")

# beta = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2)
beta = seq(0.1, 2, length.out = 20)
likelihood = NULL
for (k in 1:10){
  l = NULL
  # l = sapply(beta, estimateLikelihood, y=y)
  for (b in beta){
    temp = estimateLikelihood(b,y)
    while (temp[2]<100){
      temp = estimateLikelihood(b,y)
    }
    l = c(l, temp[1])
  }
  likelihood = rbind(likelihood, l)
}

x11()
boxplot.matrix(likelihood, xlab = "Beta values", ylab = "Log-likelihood", main = "SMC without resampling", names = round(beta, digit=2))


# Question 2
#-------------------------------------------------------------------
y = read.table("output.txt")

#beta = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2)
beta = seq(0.1, 2, length.out = 20)
likelihood = NULL
for (k in 1:10){
  l = NULL
  # l = sapply(beta, estimateLikelihood, y=y)
  for (b in beta){
    temp = estimateLikelihoodResampl(b,y)
    while (temp[2]<100){
      temp = estimateLikelihoodResampl(b,y)
    }
    l = c(l, temp[1])
  }
  likelihood = rbind(likelihood, l)
}

x11()
boxplot.matrix(likelihood, xlab = "Beta values", ylab = "Log-likelihood", main = "SMC with resampling", names = round(beta, digit=2))
