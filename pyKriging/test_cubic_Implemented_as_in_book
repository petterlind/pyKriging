import scipy.linalg as lin
import numpy as np
from matplotlib import pyplot as plt
import pdb


def d_exp(x, k, j, q):
    ''' computes the d-part of the equation (5.5) from
    elements of statistical learning
    '''
    d = ((x - k[j])**3 * (x > k[j]) -
         (x - k[q - 2])**3 * (x > k[q - 2])) / (k[q - 2] - k[j])
    return d
    

def basis_fun(x, k):
    '''
    Input:
    x - xdata, np.ndarray
    y - ydata, np.ndarray
    k - knotvector, np.ndarray
    
    Out:
    Fitting coefficients, beta
    '''
    if np.isscalar(x):
        n = 1
    else:
        n = len(x)
    q = len(knots)
    myX = np.zeros((n, q - 2))
    
    # Compute all basis function following elements of statistical learning
    dm = d_exp(x, k, q - 1, q)
    for j in range(q - 2):
        myX[:, j] = d_exp(x, k, j, q) - dm
    
    b_fun = np.hstack((np.ones((n, 1)), np.reshape(x, (n, 1)), myX))  # matrix with shape functions
    return b_fun


def speval(x, coefs, knots):
    bfun = basis_fun(x, knots)
    fun_val = np.dot(bfun, coefs)
    return fun_val


x = np.random.randn(300) * np.sqrt(2)
e = np.random.randn(300) * np.sqrt(0.5)
y = np.sin(x) + e

knots = np.linspace(np.min(x), np.max(x), num=5)

b_fun = basis_fun(x, knots)
bhat = np.linalg.lstsq(b_fun, y)[0]

x_plot = np.linspace(np.min(x), np.max(x), num=100)
y_data = speval(x_plot, bhat, knots)
plt.plot(x_plot, y_data)
plt.plot(x, y, 'r+')

# Plot the knot positions
plt.plot(knots, speval(knots, bhat, knots), 'bd')
    
plt.show()
pdb.set_trace()
