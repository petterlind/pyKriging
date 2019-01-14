import scipy.linalg as lin
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pdb


def rcs(x, y, k):
    '''
    Input:
    x - xdata, np.ndarray
    y - ydata, np.ndarray
    k - knotvector, np.ndarray
    
    Out:
    
    '''
    n = len(y)
    q = len(knots)
    myX = np.zeros((n, q - 2))
    
    for j in range(q - 2):
        # Compute all basis function
        tmp1 = (x - k[j])**3 * (x > k[j])  # compact support and positive on x>k[j]
        tmp2 = (x - k[q - 2])**3 * (x > k[q - 2]) * (k[q - 1] - k[j])  # * (k[q - 1] - k[j]) ?!
        XX = tmp1 - tmp2 / (k[q - 1] - k[q - 2])
        tmp1 = (x - k[q - 1])**3 * (x > k[q - 2])
        tmp2 = (k[q - 2] - k[j])
        XX = XX + tmp1 * tmp2 / (k[q - 1] - k[q - 2])
        myX[:, j] = XX
    
    X = np.hstack((np.ones((n, 1)), np.reshape(x, (n, 1)), myX))  # matrix with shape functions
    
    bhat = np.linalg.lstsq(X, y)[0]
    bhatt = np.zeros(len(knots) + 1)
    bhatt[len(bhat)] = (bhat[2:] * (k[0:-2] - k[-1])).sum()
    bhatt[len(bhat)] = bhatt[len(bhat)] / (k[-1] - k[-2])
    bhatt = np.hstack([bhatt, 0])
    bhatt[-1] = (bhat[2:] * (k[0:-2] - k[-2])).sum()
    bhatt[-1] = bhatt[-1] / (k[-2] - k[-1])
    bhat = np.hstack((bhat, bhatt[-2:]))
    return bhat


def speval(x, coefs, knots):
    tmp = coefs[0] + coefs[1] * x
    for k in range(len(knots)):
        tmp = tmp + coefs[k + 2] * ((x - knots[k])**3) * (x > knots[k])
    return tmp


x = np.random.randn(300) * np.sqrt(2)
e = np.random.randn(300) * np.sqrt(0.5)
y = np.sin(x) + e

knots = np.linspace(np.min(x), np.max(x), num=5)
bhat = rcs(x, y, knots)

x_plot = np.linspace(np.min(x), np.max(x), num=100)
y_data = speval(x_plot, bhat, knots)
plt.plot(x_plot, y_data)
plt.plot(x, y, 'r+')

for k in knots:
    plt.plot(k, speval(k, bhat, knots), 'bd')
    
plt.show()
pdb.set_trace()
