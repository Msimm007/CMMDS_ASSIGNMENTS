import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import *

# evaluate objective function
def eval_objfun( A, x, b, beta, L, flag="d2f" ):

    Ax = np.matmul(A,x)
    y = np.sin(Ax) - b
    
    Lx = np.matmul(L,x)
    
    # evaluate objective function
    f = 0.5*np.inner(y,y) + beta*0.5*np.inner(Lx,Lx)

    if flag == "f":
        return f

    # evaluate gradient
    AT = A.transpose()
    cosAx = np.cos(np.matmul(A,x))
    LT = L.transpose()
    LTL = np.matmul(LT,L)
    B = np.matmul(AT,np.diag(cosAx))
    
    
    df = np.matmul(B,y) + beta*np.matmul(LTL,x)
    
    if flag == "df":
        return f,df
    bh = np.matmul( np.diag(np.sin(Ax)),b) # maybe wrong dimension
    
    # evaluate hessian
    d2f = np.matmul(np.matmul(AT,np.diag(np.cos(2*Ax))),A)+ np.matmul(np.matmul(AT,np.diag(np.sin(Ax))),np.matmul(np.diag(b),A)) + beta*np.identity(n)

    return f,df,d2f;


n = 512; # problem dimension
A = np.random.rand( n, n )
x = np.random.rand( n )
L = np.identity(n)
b = np.random.rand(n)
r = np.sin(np.matmul(A,x)) - b
beta = 0.2


# initialize class
opt = Optimize();

# define function handle
fctn = lambda x, flag: eval_objfun( A, x, b, beta, L, flag )

# set objective function
opt.set_objfctn( fctn )

# perform derivative check
opt.deriv_check( x )




###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
