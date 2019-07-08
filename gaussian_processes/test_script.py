## Imports
import matplotlib.pylab as plt
import numpy as np


## Functions
def exponential_cov( x, y, params ):
    return params[0] * \
        np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

##
def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))

    return(mu.squeeze(), sigma.squeeze())

##
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new


## Visualisation Init
fig, ax = plt.subplots( nrows=5, ncols=1, figsize=(10,6) )


## Test 1
theta = [1, 10]
sigma_0 = exponential_cov(0, 0, theta)
xpts = np.arange(-3, 3, step=0.01)
ax[0].errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)
ax[0].set_ylim( -3, 3 )
ax[0].minorticks_on()
ax[0].grid( b=True, which='minor', color='0.90', \
    linestyle='--', linewidth=0.5 )
ax[0].grid( b=True, which='major', color='0.75', \
    linestyle='--', linewidth=0.75 )

x = [1.0]
y = [np.random.normal(scale=sigma_0)]
sigma_1 = exponential_cov(x, x, theta)


## Test 2
x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, theta, sigma_1, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
ax[1].errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
ax[1].plot(x, y, "ro")
ax[1].set_ylim( -3, 3 )
ax[1].minorticks_on()
ax[1].grid( b=True, which='minor', color='0.90', \
    linestyle='--', linewidth=0.5 )
ax[1].grid( b=True, which='major', color='0.75', \
    linestyle='--', linewidth=0.75 )


## Test 3
m, s = conditional([-0.7], x, y, theta)
y2 = np.random.normal(m, s)
x.append(-0.7)
y.append(y2)

sigma_2 = exponential_cov(x, x, theta)
predictions = [predict(i, x, exponential_cov, theta, sigma_2, y) for i in x_pred]
y_pred, sigmas = np.transpose(predictions)
ax[2].errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
ax[2].plot(x, y, "ro")
ax[2].set_ylim( -3, 3 )
ax[2].minorticks_on()
ax[2].grid( b=True, which='minor', color='0.90', \
    linestyle='--', linewidth=0.5 )
ax[2].grid( b=True, which='major', color='0.75', \
    linestyle='--', linewidth=0.75 )


## Test 4
x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
mu, s = conditional(x_more, x, y, theta)
y_more = np.random.multivariate_normal(mu, s)

x += x_more
y += y_more.tolist()

sigma_new = exponential_cov(x, x, theta)
predictions = [predict(i, x, exponential_cov, theta, sigma_new, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
ax[3].errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
ax[3].plot(x, y, "ro")
ax[3].set_ylim( -3, 3 )
ax[3].minorticks_on()
ax[3].grid( b=True, which='minor', color='0.90', \
    linestyle='--', linewidth=0.5 )
ax[3].grid( b=True, which='major', color='0.75', \
    linestyle='--', linewidth=0.75 )


"""
## Sinus test
N = 100
x_more = np.linspace(-3,3,N)
mu, s  = conditional(x_more, 0.0, 0.0, theta)
y_more = np.sin(x_more) + np.random.multivariate_normal(mu, s)

x += x_more
y += y_more.tolist()

sigma_new = exponential_cov(x, x, theta)
predictions = [predict(i, x, exponential_cov, theta, sigma_new, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
ax[4].errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
ax[4].plot(x, y, "ro")
ax[4].set_ylim( -3, 3 )
ax[4].minorticks_on()
ax[4].grid( b=True, which='minor', color='0.90', \
    linestyle='--', linewidth=0.5 )
ax[4].grid( b=True, which='major', color='0.75', \
    linestyle='--', linewidth=0.75 )
"""

## Show Illustrations
plt.show()
