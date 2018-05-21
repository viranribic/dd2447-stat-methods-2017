# import SVDataGenerator
#
# # generation of the parameters phi,alpha,beta
# parameters = SVDataGenerator.sv_parameter_generator(saveOpt=True, displayOpt=True, filename='myparameters.csv')
#
# # generation of a sequence of 500 log-returns
# T = 500
# yt = SVDataGenerator.sv_data_generator(parameters, T, displayOpt=True, saveOpt=True, filename='mydata.csv')
#
#
from math import sqrt,exp
from numpy.random import normal,beta
from scipy.stats import invgamma
from numpy.random import seed
import numpy as np

# The function sv_parameter_generator generates random values for the parameters phi,sigma,beta of the stochastic volatility model
#
# phi is sampled from a Beta distribution with mode=phi_mode and first shape parameter  alpha=phi_a_shape.
# sigma**2 is sampled from an inverse Gamma distribution with mode=sqrt_sigma_mode and first shape parameter alpha=sigma_shape.
# beta**2 is sampled from an inverse Gamma distribution with mode=sqrt_beta_mode and first shape parameter alpha=beta_shape.
#
# The function return the list of sampled parameters (phi,sigma,beta).
#
#
# extra arguments:
#   s=None: set s to the desired seed for numpy.random
#   saveOpt=False: set to Rrue for saving the data y_{1:T} into the file filename
#   displayOpt=True: set to True for comments
#   filename='sv_data.csv'
#
# Example: parameters=sv_parameter_generator(saveOpt=True,displayOpt=True,filename='myparameters.csv')

def sv_parameter_generator(s=None,phi_mode=0.97,phi_a_shape=300,sqrt_sigma_mode=0.20,sigma_shape=2,sqrt_beta_mode=0.20,beta_shape=2,saveOpt=False,displayOpt=True,filename='sv_parameters.csv'):
    if s is None:
        pass
    else:
        seed(s)

    phi_b_shape=2+(phi_a_shape-1)/float(phi_mode)-phi_a_shape
    phi=beta(1+phi_a_shape,1+phi_b_shape,1)[0]

    sigma_mode=sqrt_sigma_mode**2
    sigma_scale=sigma_mode*(sigma_shape+1)
    sigma=sqrt(invgamma.rvs(sigma_shape,scale=sigma_mode))

    beta_mode=sqrt_beta_mode**2
    beta_scale=beta_mode*(beta_shape+1)
    beta_=sqrt(invgamma.rvs(beta_shape,scale=beta_mode))

    if saveOpt:
        np.savetxt(filename,np.array([phi,sigma,beta_]),fmt='%.17f')
    if displayOpt:
        print 'Generated parameters:'
        print ' phi = ', phi
        print ' sigma = ',sigma
        print ' beta = ', beta_
        if saveOpt:
            print 'Saved to ',filename

    return phi,sigma,beta_


# The function sv_data_generator generates a vector y_{1:T} of observations for the stochastic volatility model for the parameters parameters=(phi,sigma,beta)
#
#  arguments:
#   parameters: [phi,alpha,beta]
#   T
#   saveOpt=False: set to Rrue for saving the data y_{1:T} into the file filename
#   displayOpt=True: set to True for comments
#   filename='sv_data.csv'
#   s=None: set s to the desired seed at numpy.random
#
# Example: yt=sv_data_generator(sv_parameter_generator(saveOpt=True,s=0),500,displayOpt=True,saveOpt=True,filename='mydata.csv')


def sv_data_generator(parameters,T,s=None,saveOpt=False,displayOpt=True,filename='sv_data.csv'):
    if s is None:
        pass
    else:
        seed(s)

    phi=parameters[0]
    sigma=parameters[1]
    beta=parameters[2]
    xt=[None for i in range(T+1)]
    yt=[None for i in range(T)]
    xt[0]=normal(0,sqrt(sigma**2/(1-phi**2)))
    for t in range(1,T+1):
        xt[t]=normal(phi*xt[t-1],sigma)
        yt[t-1]=normal(0,beta*exp(0.5*xt[t]))
    if saveOpt:
        np.savetxt(filename,np.array(yt),fmt='%.17f')
    if displayOpt:
        print 'Data generated with parameters:',parameters
        if saveOpt:
            print 'Saved to ',filename
    return yt

