import numpy as np
import sys
sys.path.append("..")

from LineSearchOpt import *
import numpy as np
from data.Data import *

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
    
    f = 0.5 * (np.linalg.norm(tanhYX-C,ord = 'fro'))**2
    
    if flag == "f":
        return f
    
    A = np.ones(tanhYX.shape) - tanhYX**2

    df = np.matmul(Y.transpose(), np.multiply(dtanhYX,tanhYX - C))
    
    if flag == "df":
        return f,df

n = 784
p = 10
m = 60000

X = np.random.rand(n, p)

# # initialize class
# opt = Optimize()

# # define function handle
# fctn = lambda X, flag: eval_objfun( Y, X, C, flag )

# # set objective function
# opt.set_objfctn(fctn)

# # perform derivative check
# opt.deriv_check(X)

""" 
Computing the derivative check separately since the LineSpace code
doesn't handle functions with Matrix valued domains.
"""

dat = Data()
[Y_train, C_train, L_train] = dat.read_mnist('train')


np.random.seed(7)
h = np.logspace( 0, -10, 10 ); # step size
v = np.random.rand(X.shape[0],X.shape[1]); # random perturbation
# evaluate objective function
f, df = eval_objfun(Y=Y_train, X=X, C=C_train, flag='df')

dfv = np.trace(df@v.T)
# vtd2fv = np.inner(v,d2f@v)
# allocate history
m = h.shape[0]
t0 = np.zeros(m)
t1 = np.zeros(m)
# t2 = np.zeros(m)

for i in range(0,m):
    fv = eval_objfun(Y=Y_train, X=x+ h[i]*v, C=C_train, flag='f')
    t0[i] = np.linalg.norm(fv - f)
    t1[i] = np.linalg.norm(fv - f - h[i]*dfv)

    print(f'h:{h[i]:.3e} tayl_poly0: {t0[i]:.3e} tayl_poly1: {t1[i]:.3e}')
# plot the errors
plt.loglog(h,t0)
plt.loglog(h,t1)

plt.legend(['t0','t1'])
plt.savefig('derive_check_3')


#
