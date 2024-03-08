import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")

from data.Data import *

import numpy as np
from LinSolve import LinSolve

class Optimize:
    def __init__(self):
        self.method = []

        # private variables
        self._ndf0 = []
        self._objfctn = []
        self._sol = LinSolve()
        self._debug = 1
        self._df_rtol = 1e-6
        self._df_atol = 1e-12
        self._maxiter = 0


    def set_objfctn( self, objfctn ):
        """
        set_objfctn set objective function (function handle);
        objective function is assumed ot be only a function of x
        i.e., the decision variable
        """
        self._objfctn = objfctn



    def set_maxiter( self, maxiter ):
        """
        set_maxiter set max number of iterations
        """
        self._maxiter = maxiter



    def set_opttol( self, tol ):
        """
        set_opttol set relative tolerance for gradient
        """
        self._df_rtol = tol


    def _get_sdir( self, x ):
        """
        _get_sdir get search direction for line search optimization
        """
        # evaluate objective function
        if self.method == "gdsc":
            f,df = self._objfctn( x, "df" )
            s = -df
        elif self.method == "newton":
            f,df,d2f = self._objfctn( x, "d2f" )
            if callable(d2f):
                s = self._sol.run_cg( d2f, -df, 1e-1, 100 )
            else:
                s = np.linalg.solve(d2f, -df)

        return s;



    def _do_linesearch( self, x, s ):
        """
        _do_linesearch do line search; implements armijo line search
        """

        # evaluate objective function
        fx, dfx = self._objfctn( x, "df" );

        # initialize flag
        success = 0;

        # set max number of linesearch iterations
        maxit = 24;

        # set initial step size
        t = 1.0;
        c = 1e-4;
        descent = np.inner( dfx, s );
        # print("descent", descent)
        # print("trial comp", fx + c * t * descent)
        
        for i in range(1,maxit):
            # evaluate objective function
            ftrial = self._objfctn( x + t*s, "f" )
            
            # print("test input", x + t*s)
            # print("f_trial", ftrial[0])

            #if self.debug:
                #print("{:e}".format(ftrial), "<", "{:e}".format(fx), "[ t =","{:e}".format(t),"]")
            
            if ftrial < (fx + c*t*descent ):
                success = 1
                break

            # divide step size by 2
            t = t / 2

        if success:
            rval = t
        else:
            rval = 0.0

        return rval



    def _check_convergence( self, x, k, df ):
        converged = 0

        ndf = np.linalg.norm( df )

        tol = self._df_rtol*self._ndf0

        if ( ndf <= tol ):
            print(">> solver converged: {:e}".format(ndf), "<", "{:e}".format(tol))
            converged = 1

        if ( ndf <= self._df_atol ):
            print(">> solver converged: {:e}".format(ndf), "<", "{:e}".format(self._ndf0))
            converged = 1

        if ( k >= self._maxiter):
            print(">> maximum number of iterations (", self._maxiter, ") reached")
            converged = 1


        return converged

    def _print_header( self, flag, reps ):

        print( reps*"-" )
        if flag == 'gdsc':
            print("executing gradient descent")
        elif flag == 'newton':
            print("executing newton's method")
        print( reps*"-" )
        print( "{:>6}".format('iter'), "{:>15}".format('||df||'), "{:>15}".format('||df||_rel'), "{:>15}".format('step') )
        print( reps*"-" )



    def run( self, x, flag="gdsc" ):
        """
        _optimize run optimizer
        """

        # set optimization method
        self.method = flag;

        f,df = self._objfctn( x, "df" )
        self._ndf0 = np.linalg.norm( df )

        reps = 55
        self._print_header( flag, reps );

        k = 0
        converged = self._check_convergence( x, k, df )

        # run optimizer
        while not converged:
            # compute search direction
            s = self._get_sdir( x )

            if np.inner( s, df ) > 0.0:
                print("not a descent direction")
                break

            # do line search
            t = self._do_linesearch( x, s )

            if t == 0.0:
                print("line search failed")
                return x
            else:
                x = x + t*s

            # check for convergence
            f,df = self._objfctn( x, "df" )

            ndf = np.linalg.norm( df )
            print("{:>6d}".format(k), "{:>15e}".format(ndf), "{:>15e}".format(ndf/self._ndf0), "{:>15e}".format(t))

            converged = self._check_convergence( x, k, df )

            k = k + 1

        print( reps*"-" )

        return x


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

def eval_objfun( Y, X, C, flag="df" ):

    YX = np.matmul(Y,X)
    # evaluate the inside of the objective function
    tanhYX, dtanhYX = tanh_actnfctn(YX, flag="df")
    
    f = 0.5 * (np.inner(tanhYX-C, tanhYX-C))
    
    if flag == "f":
        return f
    
    df = np.matmul(Y.transpose(), np.multiply(dtanhYX,tanhYX - C))
    
    if flag == "df":
        return f,df


n = 784; # problem dimension
m = 60000
p = 10

# initialize classes
opt = Optimize()
dat = Data()

[Y_train, C_train, L_train] = dat.read_mnist('train')
[Y_test, C_test, L_test] = dat.read_mnist('test')

def convert_bin_to_int(matrix):
    # Assuming matrix is a NumPy array or similar
    # The digit is determined by the index of the '1' in each row plus 1
    digits = [np.where(row == 1)[0][0] for row in matrix]
    return digits

C_train_nums = convert_bin_to_int(C_train)
C_test_nums = convert_bin_to_int(C_test)

# define function handle
fctn = lambda x, flag: eval_objfun( Y=Y_train, X=x, C=C_train_nums, flag=flag)

# initial guess
x = np.random.rand(n)

# set parameters
n_iterations = 1

opt.set_objfctn( fctn )
opt.set_maxiter( n_iterations )

# execture solver (gsc)
xgd = opt.run( x, "gdsc" )


# execture solver (newton)
# xnt = opt.run( x, "newton" )

C_pred_train = tanh_actnfctn(np.matmul(Y_train, xgd))
C_pred_test = tanh_actnfctn(np.matmul(Y_test, xgd))

def get_accuracy(y_pred, y_true):
    accuracy = np.sum(abs(y_true-y_pred) < 0.5) / len(C_train_nums)
    return accuracy


print(f"Training Accuracy for {n_iterations} iteration: ", get_accuracy(C_pred_train, C_train_nums))
print(f"Testing Accuracy for {n_iterations} iteration: ", get_accuracy(C_pred_test, C_test_nums))


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
