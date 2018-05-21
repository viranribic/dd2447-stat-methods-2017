from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy as sp
import glob
import pickle as pckl

import tensorflow as tf

%matplotlib inline
%precision 4
plt.style.use('ggplot')


class MAPEstimate:
    
    def __init__(self,opt_steps, print_stats=False):
        self.g = tf.Graph()
        
        self.opt_steps=opt_steps
        self.print_stats = print_stats
        self.s=0
        
        self.feed_dict       = None
        self.run_operations  = None
        self.run_operations_ = None
        
    def display_state(self):
        if self.feed_dict is None or self.run_operations is None: 
            raise Exception('MAPEstimate :: Not implemented')
            
    def get_opt_state(self):
        if self.feed_dict is None or self.run_operations is None: 
            raise Exception('MAPEstimate :: Not implemented')
        return self.run_operations_
    
    def optimise_graph(self):
        if self.feed_dict is None or self.run_operations is None: 
            raise Exception('MAPEstimate :: Not implemented')
        
        with self.g.as_default():
            if self.print_stats:
                print('Starting training...')
            
            # initalise session
            self.sess = tf.Session()
            self.sess.as_default()
            self.sess.run(tf.global_variables_initializer())

            for s in range(self.opt_steps):
                self.run_operations_ = self.sess.run(self.run_operations,feed_dict=self.feed_dict)
                if self.print_stats:
                    self.display_state()

            if self.print_stats:
                print('Run saved... ')

            return True
        
        
class ProbabilityFunction:
    
    def __init__(self):
        pass
    
    def rvs(self):
        pass
    
    def pdf(self,x):
        pass
    
    def cdf(self,x):
        pass
    
class TimeDepProbabilityFunction:
    
    def __init__(self,x0):
        self.x=x0
    
    def rvs(self,t):
        pass
    
    def pdf(self,t,x_t):
        pass
    
    def cdf(self,t,x_t):
        pass
    
    def update_state(self,x):
        self.x=x
        
class MultipleUniformProposals(TimeDepProbabilityFunction):
    
    def __init__(self,f_list):
        self.f_list=f_list
        self.n = len(self.f_list)
        
    def rvs(self,t):
        u = st.uniform.rvs()
        
        # interval associated to this probability
        I_low=0
        I_high=1./self.n
        for i in range(self.n-1):
            if I_low <u and u<=I_high:
                return self.f_list[i].rvs(t)
            I_low = I_high
            I_high += 1./self.n
        return self.f_list[self.n-1].rvs(t)

    
    def pdf(self,t,x_t):
        pdf = 0
        for i in range(self.n):
            pdf += self.f_list[i].pdf(t,x_t)
        return pdf
    
    def cdf(self,t,x_t):
        cdf = 0
        for i in range(self.n):
            cdf += self.f_list[i].cdf(t,x_t)
        return cdf

    
    def update_state(self,x):
        for f in self.f_list:
            f.update_state(x)
            
            
class Proposal_1(TimeDepProbabilityFunction):

    def __init__(self,x0,phi,sigma):
        TimeDepProbabilityFunction.__init__(self, x0)
        self.phi   = phi
        self.sigma = sigma
    
    def rvs(self,t):
        if t is 0:
            return st.norm.rvs(loc=0, scale=self.sigma**2/(1-self.phi**2))
        else:
            return st.norm.rvs(loc=self.phi * self.x[t-1], scale=self.sigma**2)
        
    
    def pdf(self,t,x_t):
        if t is 0:
            return st.norm.pdf(x_t,loc=0, scale=self.sigma**2/(1-self.phi**2))
        else:
            return st.norm.pdf(x_t,loc=self.phi * self.x[t-1], scale=self.sigma**2)

    
    def cdf(self,t,x_t):
        if t is 0:
            return st.norm.cdf(x_t,loc=0, scale=self.sigma**2/(1-self.phi**2))
        else:
            return st.norm.cdf(x_t,loc=self.phi * self.x[t-1], scale=self.sigma**2)
     
            
        
class Proposal_2(TimeDepProbabilityFunction):

    def __init__(self,x0,phi,sigma,T,eps=1e-16):
        TimeDepProbabilityFunction.__init__(self, x0)
        self.phi   = phi
        self.sigma = sigma
        self.T = T
        self.eps=eps
    
    def rvs(self,t):
        if t < T-1:
            return st.norm.rvs(loc=self.x[t+1]/(self.phi**2), scale=self.phi**2/(self.phi**2))
        else:
            return self.x[T-1]
        
    
    def pdf(self,t,x_t):
        if t < T-1:
            return st.norm.pdf(x_t,loc=self.x[t+1]/(self.phi**2), scale=self.phi**2/(self.phi**2))
        else:
            res = np.abs(x_t-self.x[T-1]) < self.eps
            return res.astype(float)
            #return 1. if np.abs(x_t-self.x[T-1]) < self.eps else 0.

    
    def cdf(self,t,x_t):
        if t < T-1:
            return st.norm.cdf(x_t,loc=x[t+1]/(self.phi**2), scale=self.phi**2/(self.phi**2))
        else:
            return (x_t >= self.x[T-1]).astype(float)
            #return 0. if x_t < self.x[T-1] else 1.

            
class Proposal_3(TimeDepProbabilityFunction):

    def __init__(self,x0,beta,y,eps=1e-6, x_start=0, d_x=10):
        TimeDepProbabilityFunction.__init__(self, x0)
        self.beta = beta
        self.y=y
        self.eps = eps
        self.x_start=0
        self.d_x=d_x
        
    def rvs(self,t):
        if t is 0:
            return self.x[t]
        else:
            # hyperparameter for defining the envelope distribution
            norm_sigma = 10
            x_star = np.log(self.y[t]**2/(self.beta**2)) # maximum of the given function
            f_star = self._f(t,x_star)
            M = f_star/( 1/np.sqrt(2*np.pi*norm_sigma**2)) # scale max of the gaussian to be over the max of given function
            # M is the constant in inequality: Mq(x) >= p(x) 
            
            while True:
                u = st.uniform.rvs()
                
                x_sample = st.norm.rvs(loc=x_star,scale=norm_sigma)
                y_t = self._f(t,x_sample) # y of target
                y_g = M*st.norm.pdf(x_sample,loc=x_star,scale=norm_sigma) # y of gaussian 
                
                if u <= y_t / y_g:
                    return x_sample
        
    def pdf(self,t,x_t):
        if t is 0:
            res = np.abs(x_t-self.x[0]) < self.eps
            return res.astype(float)
            
            return 1. if abs(x_t-self.x[0]) < self.eps else 0.
        else:
            return self._f(t,x_t)

    def cdf(self,t,x_t):
        if t is 0:
            #return 0. if x_t < self.x[0] else 1.
            return (x_t >= self.x[0]).astype(float)
        else:
            return sp.integrate.quad(lambda x: self._f(t,x),np.NINF, x_t)[0] # the return is the probabiliy and error
            
    def _f(self,t,x_t):
        C = 2* np.abs(self.y[t])/(self.beta * np.sqrt(2*np.pi))
        power_1 = - self.y[t]**2/(2*self.beta**2) * np.exp(-x_t)
        power_2 = -x_t/2

        return C * np.exp(power_1 + power_2)

    
class LikelihoodFunction(ProbabilityFunction):
    
    def __init__(self,phi,sigma,beta,yt):
        self.phi=phi
        self.sigma=sigma
        self.beta=beta
        self.yt = yt
        
    def rvs(self):
        raise Exception('LikelihoodFunction :: rvs not implemented')
        
    
    def pdf(self,x):
        p = st.norm.pdf(x[0], loc=0, scale= self.sigma**2/(1-self.phi**2))
        
        for t in range(1,len(x)) :
            p*=st.norm.pdf(x[t], loc= self.phi * x[t-1], scale= self.sigma**2)
            p*=st.norm.pdf(yt[t-1], loc=0, scale= self.beta**2*np.exp(x[t]))
        return p
    
    def cdf(self,x):
        raise Exception('LikelihoodFunction :: cdf not implemented')
        
        
        
        
class MCMC_sampler:
    
    def __init__(self,x0,g,p,T,N_bi):
        self.x=x0
        self.g=g
        self.p=p
        self.T=T
        self.sample(N_bi)
        
    def sample(self,N):t
        x_current = self.x
        #prepare a samples list
        samples = []

        # for burn_in + N samples repeat
        for s in range(N):
            # sample from g
            t = int(st.uniform.rvs() * T)
            x_t = g.rvs(t)

            x_next = np.copy(x_current)
            x_next[t] = x_t

            p_c = p.pdf(x_current)
            p_n = p.pdf(x_next)
            alpha = p_n/p_c
            r=min(1,alpha)

            # include sample
            u = st.uniform.rvs()
            if u < r:
                x_current = x_next
                g.update_state(x_current)
            
            samples.append(x_current )
                
        # update the last state to the new state
        self.x = x_current
        return samples