from cvxopt import matrix, spmatrix, exp, sin, cos
import numpy as np

## Zero prior
def pzero( x ):
    x = matrix(x)
    return matrix( 0.0, x.size )

## Sine prior
def psin( x ):
    x = matrix(x)
    return sin(x)

## Cosine prior
def pcos( x ):
    x = matrix(x)
    return cos(x)

## Sine-Cosine prior
def psincos( x ):
    x = matrix(x)
    return sin(x) + cos(x)

## User-specified prior
def puserspec( x , prior=lambda x: 0.0*x ):
    x = matrix(x)
    return prior(x)
