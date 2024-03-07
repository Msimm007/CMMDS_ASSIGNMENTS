import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import *
import numpy as np

def tanh_actnfctn( x, flag="f" ):
    # create vector of ones
    e = np.ones( x.shape )

    # evaluate function
    f = np.tanh( x )

    if flag == "f":
        return f

    # evaluate gradient
    df  = 1 - f**2

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = -2*f + 2*(f**3)

    return f, df, d2f

# evaluate objective function
def eval_objfun( Y, X, C, p, flag="d2f" ):

    YX = np.matmul(Y,X)
    
    # evaluate the inside of the objective function
    sigma = np.tanh(YX)
    
    f = 0.5 * np.trace(np.matmul(np.transpose(sigma - C), sigma))

    if flag == "f":
        return f


    # A = np.matmul(Y,np.identity(p)-np.tanh(YX))   
        
    # evaluate gradient
    # df = np.matmul(A,f) 
    
    tanhYX, dtanhYX = tanh_actnfctn(YX, flag="df")

    A = np.multiply(tanhYX - C, dtanhYX)

    df_matrix = np.matmul(Y.transpose(), A)
    
    df = df_matrix.reshape(-1)

    if flag == "df":
        return f,df

    # evaluate hessian
    # d2f = 0.5*(Y + Y.transpose())
    d2f = np.identity(n)
    
    return f,df,d2f

#n = 28*28;
n = 32
p = 10
#m = 60000
m = 600


Y = np.random.rand(m, n)
C = np.random.rand(m,p)
X = np.random.rand(n,p)



# initialize class
opt = Optimize()

# define function handle
fctn = lambda X, flag: eval_objfun( Y, X, C, p, flag )

# set objective function
opt.set_objfctn(fctn)

# perform derivative check
opt.deriv_check(X)




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
#
