import scipy.linalg as lin
import numpy as np
from matplotlib import pyplot as plt
import pdb
import pyKriging
from pyKriging import matrixops
from pyKriging import samplingplan
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools


def d_exp(x, k, j, q):
    ''' computes the d-part of the equation (5.5) from
    elements of statistical learning
    '''
    d = ((x - k[j])**3 * (x > k[j]) -
         (x - k[q - 2])**3 * (x > k[q - 2])) / (k[q - 2] - k[j])
    return d
    
    
def basis_1d(x, k):
    '''
    Input:
    x - xdata, np.ndarray
    y - ydata, np.ndarray
    k - knotvector, np.ndarray
    
    Out:
    basis value, 1d
    '''
    
    if np.isscalar(x):
        n = 1
    else:
        n = len(x)
    q = len(k)
    myX = np.zeros((n, q - 2))
    
    # Compute all basis function following elements of statistical learning
    dm = d_exp(x, k, q - 1, q)
    for j in range(q - 2):
        myX[:, j] = d_exp(x, k, j, q) - dm
    
    b_fun = np.hstack((np.ones((n, 1)), np.reshape(x, (n, 1)), myX))  # matrix with shape functions
    return b_fun
    
def basis_2d(x, k=None, nd=2):
    ''' nd-basis for the spline'''
    n = x.shape[0]
    
    def compute_basis(b_mat, nd, n, level):
        ''' nested loop that computes nd-outer product of basis vectors!'''
        A = []
        res = 1
        lists = []
        for num in level:
            lists.append(range(0, num)) 
            
        nr = list(x for x in itertools.product(*lists))
        
        for ind in nr:
            res = 1
            for d, i in zip(range(nd), ind):
                res = res * b_mat[d, i]
            A.append(res) 
        return A
        
    if np.isscalar(x[0]): # if scalar create an array!
        x = np.asarray([x])
        nr_p = 1
    else:
        nr_p  = len(x[:, 0])
    
    level = np.asarray([3] * nd) # starting with a 3*3 cube
    while np.prod(level) < n / 2:
        
        # Check if all are the same
        if (level == level[0]).all():
            level[0] = level[0] + 1
        
        else: # find first value smaller then level[0]
            for x, val in enumerate(level):
                if val < level[0]:
                    level[x] = val + 1
    # Compute all the basis function values
    basis_mat = []
    # basis_mat = np.zeros((nd, nr_p, 3)) #len(k))) # storage structure!
    
    for i in range(nd):
        basis_mat.append(basis_1d(x[:, i], np.linspace(-1, 1, level[i])))  # Same knot vector in both directions
    basis_mat = np.asarray(basis_mat)
    
    # Assemble, following Nils Carlsson's Master thesis
    B_row = None
    
    for i in range(nr_p):
        A_conc = np.asarray(compute_basis(basis_mat[:,i,:], nd, n, level))
        
        if B_row is None:
            B_row = A_conc
        else:
            B_row = np.append(B_row, A_conc)  # one long row of data, Rewrite for speed?
    F = np.reshape(B_row, (-1, len(A_conc)))  # reshape into matrix
    
    if F is None:
        raise(TypeError)
        print('F cant be None!')
    
    return F
    
    
# def basis_2d(x, k):
#     '''
#     Input:
#     x - xdata, np.ndarray
#     y - ydata, np.ndarray
#     k - knotvector, np.ndarray
# 
#     Out:
#     F - basis value matrix, 2d
#     '''
#     if not np.isscalar(x[0]): #
        # b_fun_u = basis_1d(x[:, 0], k)  # Same knot vector in both directions
        # b_fun_v = basis_1d(x[:, 1], k)
#         length = len(x[:, 0])
# 
#         # Do the matrix for the surface following Nils Carlsson's MT
#         B_row = None
#         for i in range(length):
#             A_conc = np.concatenate(np.outer(b_fun_u[i], b_fun_v[i]))  # Same knot vector in both directions
#             if B_row is None:
#                 B_row = A_conc
#             else:
#                 B_row = np.append(B_row, A_conc)  # one long row of data, Rewrite for speed?
#         F = np.reshape(B_row, (-1, len(A_conc)))  # reshape into matrix
# 
#     else:
#             [b_fun_u] = basis_1d(x[0], k)
#             [b_fun_v] = basis_1d(x[1], k)
#             F = np.concatenate(np.outer(b_fun_u, b_fun_v)) # The only part that needs to be changed for higher order applications!?
#     pdb.set_trace()        
#     return F
# 
    
def speval(x, coefs, knots):
    bfun = basis_2d(x, knots)
    fun_val = np.dot(bfun, coefs)
    return fun_val


if __name__ == '__main__':
    num_p = 100
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    sp = samplingplan(2)
    X = sp.rlh(num_p)
    testfun = pyKriging.testfunctions().branin
    y = testfun(X)
    knots = np.linspace(0, 1, num=20)

    b_fun = basis_2d(X, knots)
    bhat = np.linalg.lstsq(b_fun, y)[0]

    # Plot
    X_p, Y_p = np.meshgrid(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05))
    zs = np.array([speval([x, y], bhat, knots) for x, y in zip(np.ravel(X_p), np.ravel(Y_p))])
    Z = zs.reshape(X_p.shape)

    # real function
    z_real = np.array([testfun(np.array([x, y])) for x, y in zip(np.ravel(X_p), np.ravel(Y_p))])
    Z_r = z_real.reshape(X_p.shape)


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    ax.plot_surface(X_p, Y_p, Z, cmap=cm.coolwarm)
    ax.plot_wireframe(X_p, Y_p, Z_r)

    ax.scatter(X[:, 0], X[:, 1], y)

    plt.show()
    pdb.set_trace()
