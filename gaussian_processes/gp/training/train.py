from cvxopt import matrix


def train( x, m, k, **kwargs ):
    x = matrix(x)

    ## Mean
    mu    = matrix( m(x) )

    ## Co-variance
    Sigma = matrix( 0.0, (x.size[0],x.size[0]) )
    for i in range(x.size[0]):
        for j in range(i,x.size[0]):
            co_var     = k( x[i], x[j], **kwargs )
            Sigma[i,j] = co_var
            Sigma[j,i] = co_var

    ## Return statement
    return mu, Sigma
