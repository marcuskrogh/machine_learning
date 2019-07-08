## Imports
from cvxopt import matrix, exp

## Squared exponential kernel
def sqexp( x, y, lambda_=1.0 ):
    x = matrix(x)
    y = matrix(y)
    xmy = x - y
    return exp( -1/(2*lambda_**2)*xmy.T*xmy )
