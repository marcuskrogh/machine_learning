## Imports
from cvxopt import matrix, spmatrix, exp, sin, cos, sqrt, min
import numpy as np

## Prediction method for Gaussian processes
def predict(                             \
        ## Prediction data and training data
        x_, x, y,                        \
        ## Trained model
        mu, Sigma,                       \
        ## Prior and Kernel
        prior, kernel,                   \
        ## Hyper-parameters (tuned)
        lambda_=1.0, beta=1.0, sigma=0.0 ):
    ## Type-check x_
    x_ = matrix(x_)
    I_x  = matrix( spmatrix( 1.0, range(mu.size[0]), range(mu.size[0]) ) )
    I_x_ = matrix( spmatrix( 1.0, range(x_.size[0]), range(x_.size[0]) ) )

    ## Pre-allocate
    K_xx_  = matrix( 0.0, (x.size[0],x_.size[0]) )
    K_x_x_ = matrix( 0.0, (x_.size[0],x_.size[0]) )

    ## Compute Co-variances
    for i in range(x.size[0]):
        for j in range(x_.size[0]):
            K_xx_[i,j] = kernel( x[i,:].T, x_[j,:].T, lambda_=lambda_ )

    for i in range(x_.size[0]):
        K_x_x_[i,i] = kernel( x_[i,:].T, x_[i,:].T, lambda_=lambda_ )

    ## Prediction
    Sigma_inv = matrix( np.linalg.inv(Sigma + sigma**2*I_x) )
    mu_x_     = prior(x_)*beta + K_xx_.T*Sigma_inv*(y - mu*beta)
    Sigma_x_  = K_x_x_ + sigma**2*I_x_ - K_xx_.T*Sigma_inv*K_xx_

    return mu_x_, Sigma_x_
