IBP = function(N, alpha){
  
  # Generate initial number of dishes from a Poisson(alpha)
  n0 = rpois(1,alpha)
  K = n0
  Z = matrix(0, nrow = N, ncol = n0)
  Z[1,] = 1
  
  # Count the number of non-zero entries per each column k
  m = NULL
  for (k in 1:dim(Z)[2])
    m = c(m, sum(Z[,k]))
  
  for (i in 2:N){
    
    # Compute probability of visiting past dishes:
    # the i-th customer tries each previously sampled dish
    # with probability proportional to the number of people
    # that have already tried the dish
    prob = m/i
    
    # Metropolis-Hastings step
    rand = runif(K)
    idx = prob > rand
    Z[i,] = as.numeric(idx)
    
    # Compute the number of new dishes visited by customer i
    k_new = rpois(1,alpha/i)
    
    if (k_new > 0){
      Z_new = matrix(0, nrow = N, ncol = k_new)
      Z = cbind(Z, Z_new)
      new_dishes = seq(K+1, K+k_new)
      Z[i, new_dishes] = 1
    }
    
    # Update matrix size and dish popularity count
    K = K + k_new
    m = NULL
    for (k in 1:dim(Z)[2])
      m = c(m, sum(Z[,k]))
  }
  
  return (Z)
  
}

update_Z = function(Z, W, alpha){
  
  sigma = 0.5
  N = dim(Z)[1]
  K = dim(Z)[2]
  
  # Count the number of non-zero entries per each column k
  m = NULL
  for (k in 1:K)
    m = c(m, sum(Z[,k]))
  
  for (i in 1:N){
    
    # Update z_ik
    # for (k1 in 1:K){
    #   p_new = m[k1]
    #   for (j in 1:N){
    #     s = 0
    #     for (k2 in 1:K){
    #       if (k2!=k1 & j!=i)
    #         s = s + Z[i,k1]*Z[j,k2]*W[k1,k2]
    #     }
    #     p_new = p_new * (1/(1+exp(-s)))
    #   }
    #   print(p_new)
    #   rand = runif(1)
    #   if (rand < min(1,p_new)){
    #     Z[i,k1] = 1
    #   } else {
    #     Z[i,k1] = 0
    #     print("z is 0")
    #   }
    # }
    
    idx = NULL
    for (k in 1:K){
      m_ik = m[k] - Z[i,k]
      if (m_ik > 0){
        p = (m_ik+alpha/K)/(N+alpha/K)
        rand = runif(1)
        if (rand < p){
          Z[i,k] = 1
        } else {
          Z[i,k] = 0
        }
      } else {
        idx = c(idx,k)
      }
    }
    
    if (length(idx) > 0){
      Z = Z[,-idx]
      W = W[-idx,-idx]
      N = dim(Z)[1]
      K = dim(Z)[2]
    }
    
    # Update the number of dishes
    k_new = rpois(1,alpha/N)
    if (k_new > 0){
      Z_new = matrix(0, nrow = N, ncol = k_new)
      Z = cbind(Z, Z_new)
      new_dishes = seq(K+1, K+k_new)
      Z[i, new_dishes] = 1
      
      W_new = matrix(0, nrow = K + k_new, ncol = K + k_new)
      W_new[1:K, 1:K] = W
      W = W_new
      
      for (k1 in (K+1):(K+k_new)){
        for (k2 in (K+1):(K+k_new)){
          if (k1!=k2)
            W[k1,k2] = rnorm(1, 0, sigma)
          else
            W[k1,k2] = 10
        }
      }
      
      K = K + k_new
      m = NULL
      for (k in 1:K)
        m = c(m, sum(Z[,k]))
    }
    
  }
  
  return (list(Z, W))
  
}

update_W = function(Z, W){
  
  sigma = 0.5
  
  # Drop all weights that correspond to all-zero features
  idx = NULL
  for (k2 in 1:dim(Z)[2]){
    if (all(Z[,k2] == 0))
      idx = c(idx, k2)
  }
  
  if (length(idx)>0){
    W = W[-idx,-idx]
    Z = Z[,-idx]
  }
    
  for (k2 in 1:dim(Z)[2]){
    for (k1 in 1:dim(Z)[2]){
        
      # Metropolis-Hastings step
      w_new = rnorm(1, W[k1,k2], sigma)
      w_old = W[k1,k2]
      new_contr = 1
      old_contr = 1
      for (i in 1:dim(Z)[1]){
        for (j in 1:dim(Z)[1]){
          new_contr = new_contr * (1/(1+exp(-Z[i,k1]*Z[j,k2]*w_new)))
          old_contr = old_contr * (1/(1+exp(-Z[i,k1]*Z[j,k2]*w_old)))
        }
      }
      acc_p = (new_contr/old_contr) * (dnorm(w_new, 0, sigma)/dnorm(w_new, 0, sigma)) *(dnorm(w_old, mean = w_new, sd = sigma)/dnorm(w_new, mean = w_old, sd = sigma))  
      r = min(acc_p, 1)
      rand = runif(1)
      if (rand < r)
        W[k1,k2] = w_new
    }
  }
  
  return (list(Z, W))
  
}