## Imports
from cvxopt import matrix, sqrt
import numpy as np

## Default output function
def outfun_( x ):
    x = matrix( x )
    return matrix( 0.0, (x.size[0],1) )

## Generation of data with user-speficied output function
def generate_data( xmin, xmax, N, outfun=outfun_, noise=0.0 ):
    ## Data space
    xint = xmax - xmin
    x    = (( matrix(range(N+1))/N )*xint - 0.5*xint)

    ## Output data
    e_k = matrix( sqrt(noise) * np.random.randn(N+1,1) )
    y_true = outfun(x)

    y      = y_true + e_k

    ## Return statement
    return x, y, y_true
