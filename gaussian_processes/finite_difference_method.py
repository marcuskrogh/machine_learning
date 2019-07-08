## Imports
from cvxopt import matrix, spmatrix, sqrt


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++ Approximation Functions ++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## Forward difference approximation
def forward_difference( fun, x, p, ):
    dx = sqrt(sum(p**2))
    return ( fun(x + p) - fun(x) ) / dx

## Backward difference approximation
def backward_difference( fun, x, p, ):
    dx = sqrt(sum(p**2))
    return ( fun(x) - fun(x - p) ) / dx

## Central difference approximation
def central_difference( fun, x, p, ):
    dx = sqrt(sum(p**2))
    d1 = fun(x + p)
    d2 = fun(x - p)
    return ( d1 - d2 ) / (2*dx)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++ Finite difference method ++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## Finite difference method - Devirative approximation
def fdm( fun, x, jacobian=False, method=forward_difference ):
    ## Type checking
    x = matrix( x )
    f = matrix( fun( x ) )

    ## Size definitions
    Nf, _ = f.size
    Nx, _ = x.size

    ## Define optimal step size
    u   = 1.1e-16
    eps = sqrt( u )

    ## Pre-allocation of derivative
    dfdx = matrix( 0.0, (Nx,Nf) )

    ## Central difference approximation
    I = matrix( spmatrix( 1.0, range(Nx), range(Nx) ) )
    for nx in range(Nx):
        p = I[:,nx]*eps
        dfdx_ = matrix( method( fun, x, p ) )
        for nf in range(Nf):
            dfdx[nx,nf] = dfdx_[nf]

    ## Return statement
    if jacobian:
        return dfdx.T
    else:
        return dfdx
