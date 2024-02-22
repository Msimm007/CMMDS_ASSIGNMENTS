import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import *

# evaluate objective function
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

    return f,df,d2f


n = 512 # problem dimension
Q = np.random.rand(n, n)
x = np.random.rand(n)
b = np.random.rand(n)
c = np.random.uniform(-10.0, 10.0)

# initialize class
opt = Optimize()

# define function handle
fctn = lambda x, flag: eval_objfun( Q, x, b, c, flag )

# set objective function
opt.set_objfctn(fctn)

# perform derivative check
opt.deriv_check(x)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
#
