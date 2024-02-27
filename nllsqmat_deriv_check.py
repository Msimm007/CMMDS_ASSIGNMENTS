import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import *

# evaluate objective function
def eval_objfun( Y, X, C, p, flag="d2f" ):

    YX = np.matmul(Y,X)
    
    # evaluate objective function
    f = np.tanh(YX) - C

    if flag == "f":
        return f


    A = np.matmul(Y,np.identity(p)-np.tanh(YX))    
    
    # evaluate gradient
    df = np.matmul(A,f) 

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = 0.5*(Y + Y.transpose())

    return f,df,d2f


#n = 28*28;
n = 32
p = 10
#m = 60000
m = 600


Y = np.random.rand(m, n)
C = np.random.rand(m,p)
Xtrue = np.random.rand(n,p)



# initialize class
opt = Optimize()

# define function handle
fctn = lambda X, flag: eval_objfun( Y, X, C, p, flag )

# set objective function
opt.set_objfctn(fctn)

# perform derivative check
opt.deriv_check(Xtrue)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
#
