import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")

from LineSearchOpt import *
from data.Data import *

import numpy as np

def matrix_to_digits(matrix):
    digits = [np.where(row == 1)[0][0] for row in matrix]
    return digits

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
def eval_objfun( Y, X, C, flag="df" ):

    YX = np.matmul(Y,X)
    
    # evaluate the inside of the objective function
    tanhYX, dtanhYX = tanh_actnfctn(YX, flag="df")
    
    f = 0.5 * np.inner(tanhYX - C, tanhYX-C)

    if flag == "f":
        return f


    # A = np.matmul(Y,np.identity(p)-np.tanh(YX))   
        
    # evaluate gradient
    # df = np.matmul(A,f) 
    
    A = np.multiply(tanhYX - C, dtanhYX)

    df = np.matmul(Y.transpose(), A)
    
    if flag == "df":
        return f,df

    # evaluate hessian
    # d2f = 0.5*(Y + Y.transpose())
    d2f = np.identity(n)
    
    return f,df,d2f


n = 784; # problem dimension
m = 60000
p = 10

# initialize classes
opt = Optimize()
dat = Data()

[Y_train, C_train, L_train] = dat.read_mnist('train')
[Y_test, C_test, L_test] = dat.read_mnist('test')

C_train = matrix_to_digits(C_train)
C_test = matrix_to_digits(C_test)


# define function handle
fctn = lambda x, flag: eval_objfun( Y_train, x, C_train, flag=flag)

# initial guess
x = np.zeros( n )

# set parameters
n_iterations = 100
opt.set_objfctn( fctn )
opt.set_maxiter( n_iterations )

# execture solver (gsc)
xgd = opt.run( x, "gdsc" )

# execture solver (newton)
# xnt = opt.run( x, "newton" )

C_pred_train = tanh_actnfctn(np.matmul(Y_train, xgd))
C_pred_test = tanh_actnfctn(np.matmul(Y_test, xgd))

def get_accuracy(y_pred, y_true):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print(f"Training Accuracy for {n_iterations} iteration: ", get_accuracy(C_pred_train, C_train))
print(f"Testing Accuracy for {n_iterations} iteration: ", get_accuracy(C_pred_test, C_test))


z = np.linspace( 0, 1, n)
plt.plot( z, xgd, marker="1", linestyle='', markersize=12)
# plt.plot( z, xnt, marker="2", linestyle='', markersize=12)
# plt.plot( z, C_test )
plt.legend(['gradient descent', 'newton', r'$x^\star$'], fontsize="10")
plt.savefig("mnist_ggd")



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################
