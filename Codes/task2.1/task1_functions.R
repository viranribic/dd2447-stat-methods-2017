normalSamplingX = function(mu, sigma){
  return (rnorm(1, mu, sigma))
}

estimateLikelihood = function(beta, y){
  
  T = 500
  N = 500
  phi = 0.9709
  sigma = 0.1315
  l = 0 
  
  # vector_w = NULL
  
  # Time t = 1, thanks to the stationarity setting
  x0 = rnorm(N, 0, sqrt(sigma^2/(1-phi^2)))
  
  # Sampling N times, looking at x0 values
  x_old = sapply(phi*x0, normalSamplingX, sigma=sigma)
  x_new = x_old
  
  p1 = sum(mapply(dnorm, x_old, phi*x0, sd=sigma) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))

  alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_old))) * p1
  w = alpha
  W = w/sum(w)
  
  l = l + log(sum(w))

  # At each time, first sample N points, then compute the weights, that give
  # us the estimate of the target probability distribution
  for (t in 2:T){
    x_new = sapply(phi*x_old, normalSamplingX, sigma = sigma)

    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_new)))
    
    if (is.nan(l+log(sum(W*alpha)))){
      print(t)
      break
    }
    
    l = l + log(sum(W*alpha))

    w = w * alpha
    W = w/sum(w)
    
    x_old = x_new
  }

  return(c(l,t))
  
}

generateU = function(v, u1, N){
  return (u1+(v-1)/N)
}

estimateLikelihoodResampl = function(beta, y){
  
  T = 500
  N = 500
  phi = 0.9709
  sigma = 0.1315
  l = 0
  
  u = array(0,N)
  u[1] = runif(1, 0, 1/N)
  u[2:N] = sapply(2:N, generateU, u1=u[1], N=N)
  
  # Time t = 1
  x0 = rnorm(N, 0, sqrt(sigma^2/(1-phi^2)))
  
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
  
  if (!is.null(x_new)){
    
    p1 = sum(mapply(dnorm, x_new, phi*x0, sd=sqrt((sigma^2/(1-phi^2)))) * dnorm(x0,  0, sqrt(sigma^2/(1-phi^2))))
    alpha = dnorm(y[1,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = alpha*p1
    W_new = w_new/sum(w_new)
    l = l + log(sum((1/N)*alpha))
  
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
    
    alpha = dnorm(y[t,1], 0, sqrt((beta^2)*exp(x_new)))
    w_new = w_new*alpha
    W_new = w_new/sum(w_new)
    l = l + log(sum((1/N)*alpha))

  }
  
  return(c(l,t))
  
}