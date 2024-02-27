import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from LineSearchOpt import *
from data.Data import *

"""The following is from quadobj_deriv_check. We could just import
    this but I'm following Mang's convention in his examples.
"""

def eval_objfun( Q, x, b, c, flag="d2f" ):

# evaluate objective function
    Qx = np.matmul(Q, x)

    f = 0.5*np.inner(x,Qx) + np.inner(b,x) + c


    if flag == "f":
        return f

    # evaluate gradient
    QT = Q.transpose()
    df = 0.5*np.inner(Q + QT,x) + b

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = 0.5*(Q + QT)

    return f,df,d2f;


n = 512 # problem dimension
Q_random = np.random.rand(n, n)
x = np.random.rand(n)
b_term = np.random.rand(n)
c = np.random.uniform(-10.0, 10.0)

# initialize class
opt = Optimize()

# define function handle
fctn = lambda x, flag: eval_objfun( Q_random, x, b_term, c, flag )

# set objective function
opt.set_objfctn(fctn)

# perform derivative check
opt.deriv_check(x)


"""
Convexity checks for random Q.
"""

# define function handle
fctn = lambda x, flag: eval_objfun( Q_random, x, b_term, 0.03, flag )
opt.set_objfctn( fctn )
m = 100
bound = np.zeros(2)
t = np.linspace( bound[0], bound[1], m )
plt.plot( t, g )
plt.show()
plt.savefig("cvx_check_random_Q")

""" 
Convexity Check for SPD Q
"""

data = Data()
Q_spd = data.get_spd_mat(n)

# define function handle
fctn = lambda x, flag: eval_objfun( Q_spd, x, b_term, 0.03, flag )
opt.set_objfctn( fctn )

bound = np.zeros(2)
b = 5

bound[0] = -b # lower bound
bound[1] =  b # upper bound
m = 100 # number of samples

# number of random perturbations
ntrials = 10

g = np.zeros([m,ntrials])

# draw random perturbations
for i in range(ntrials):
    # draw a random point
    x = np.random.rand( n )
    # compute 1d function along line: g(t) = f( x + t v )
    g[:,i] = opt.cvx_check( x, bound, m )


# plot
t = np.linspace( bound[0], bound[1], m )
plt.plot( t, g )
plt.show()
plt.savefig("cvx_check_spd_Q")