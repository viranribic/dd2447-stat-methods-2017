normalSamplingX = function(mu, sigma){
  return (rnorm(1, mu, sigma))
}

update_x = function(sigma2, phi, T){
  x = array(0,T+1)
  x[1] =  rnorm(1, 0, sqrt(sigma2/(1-phi^2)))
  
  for (t in 2:(T+1))
    x[t] = rnorm(1, phi*x[t-1], sqrt(sigma2))
  
  return (x)
}

update_sigma2 = function(x, phi, T){
  a = 0.01
  b = 0.01
  
  temp = sum((x[2:length(x)]-phi*x[1:(length(x)-1)])^2)
  
  return (rinvgamma(1, shape=a+T/2, scale=b+temp/2))
  # return (rinvgamma(1, a+T/2, b+temp/2))
}

update_beta2 = function(x, y, T){
  a = 0.01
  b = 0.01
  
  x = x[-1]
  temp = sum(exp(-x)*(y^2))
  
  # temp = 0
  # for (t in 1:T)
  #   temp = temp + exp(-x[t])*y[t]^2
  
  return (rinvgamma(1, shape=a+T/2, scale=b+temp/2))
  # return (rinvgamma(1, a+T/2, b+temp/2))
}

samplingIG = function(a, b){
  return (rinvgamma(1, shape=a, scale=b))
}




generateU = function(v, u1, N){
  return (u1+(v-1)/N)
}

smc_tot = function(sigma, phi, beta, T){
  N = 1000
  X = NULL # N columns, T+1 rows --> a particle path is a column
  
  u = array(0,N)
  u[1] = runif(1, 0, 1/N)
  u[2:N] = sapply(2:N, generateU, u1=u[1], N=N)
  
  # Time t = 1
  x0 = rnorm(N, 0, sqrt(sigma^2/(1-phi^2)))
  X = rbind(X, x0)
  
  # Sampling N times, looking at x0 values
  x_old = sapply(phi*x0, normalSamplingX, sigma=sigma)

  p1 = sum(mapply(dnorm, x_old, phi*x0, sd=sigma) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))
  alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_old))) * p1
  w_old = alpha
  W_old = w_old/sum(w_old)
  
  offspring = array(0,length(w_old))
  x_new = NULL
  for (i in 1:N){
    if (i==1)
      lower = 0
    else 
      lower = sum(W_old[1:(i-1)])
    upper = sum(W_old[1:i])
    count = length(which(u>=lower & u<=upper))
    offspring[i] = count
    if (count > 0){
      x_new = c(x_new, rep(x_old[i], offspring[i])) 
    }
  }
  
  X = rbind(X, x_new)
  
  if (!is.null(x_new)){
    
    p1 = sum(mapply(dnorm, x_new, phi*x0, sd=sigma) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))
    alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = alpha*p1
    W_new = w_new/sum(w_new)

  }
  
  for (t in 2:T){
    
    x_old = sapply(phi*x_new, normalSamplingX, sigma = sigma)

    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_old)))
    w_old = w_new * alpha
    W_old = w_old/sum(w_old)
    
    u = array(0,N)
    u[1] = runif(1, 0, 1/N)
    u[2:N] = sapply(2:N, generateU, u1=u[1], N=N)
    
    # Resampling step
    x_new = NULL
    offspring = array(0,length(w_old))
    for (i in 1:length(offspring)){
      if (i==1)
        lower = 0
      else 
        lower = sum(W_old[1:(i-1)])
      upper = sum(W_old[1:i])
      count = length(which(u>=lower & u<=upper))
      offspring[i] = count
      if (count > 0){
        x_new = c(x_new, rep(x_old[i], offspring[i])) 
      }
    }
    
    if (is.null(x_new)){
      print(t)
      break
    }
    
    X = rbind(X, x_new)
    
    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = w_new*alpha
    W_new = w_new/sum(w_new)
    
  }
  
  return(X)
  
}

smc = function(Xk, sigma, phi, beta, T, j){
  N = 1000
  k = N
  output = NULL
  
  u = array(0,N)
  u[1] = runif(1, 0, 1/N)
  u[2:N] = sapply(2:N, generateU, u1=u[1], N=N)
  
  # Time t = 1
  x0 = rnorm(N, 0, sqrt(sigma^2/(1-phi^2)))
  output = c(output, x0[j])
  
  # Sampling N times, looking at x0 values
  x_old = sapply(phi*x0, normalSamplingX, sigma=sigma)
  x_old[k] = Xk[1]
  
  p1 = sum(mapply(dnorm, x_old, phi*x0, sd=sigma) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))
  alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_old))) * p1
  w_old = alpha
  W_old = w_old/sum(w_old)
  
  offspring = array(0,length(w_old))
  x_new = NULL
  for (i in 1:N){
    if (i==1)
      lower = 0
    else 
      lower = sum(W_old[1:(i-1)])
    upper = sum(W_old[1:i])
    count = length(which(u>=lower & u<=upper))
    offspring[i] = count
    if (count > 0){
      x_new = c(x_new, rep(x_old[i], offspring[i])) 
    }
  }
  
  x_new[k] = Xk[1] 
  
  if (!is.null(x_new)){
    
    p1 = sum(mapply(dnorm, x_new, phi*x0, sd=sigma) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))
    alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = alpha*p1
    W_new = w_new/sum(w_new)
    
    output = c(output, x_new[j])

  }
  
  for (t in 2:T){
    
    x_old = sapply(phi*x_new, normalSamplingX, sigma = sigma)
    x_old[k] = Xk[t] 
    
    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_old)))
    w_old = w_new * alpha
    W_old = w_old/sum(w_old)
    
    u = array(0,N)
    u[1] = runif(1, 0, 1/N)
    u[2:N] = sapply(2:N, generateU, u1=u[1], N=N)
    
    # Resampling step
    x_new = NULL
    offspring = array(0,length(w_old))
    for (i in 1:length(offspring)){
      if (i==1)
        lower = 0
      else 
        lower = sum(W_old[1:(i-1)])
      upper = sum(W_old[1:i])
      count = length(which(u>=lower & u<=upper))
      offspring[i] = count
      if (count > 0){
        x_new = c(x_new, rep(x_old[i], offspring[i])) 
      }
    }
    
    if (is.null(x_new)){
      print(t)
      break
    }
    
    x_new[k] = Xk[t] 
    
    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = w_new*alpha
    W_new = w_new/sum(w_new)
    
    output = c(output, x_new[j])

  }
  
  return(output)
  
}