from pyKriging import samplingplan
import pyKriging.testfunctions as testfun
import scipy
import numpy as np
import pdb
from matplotlib import pyplot as plt


def least_square(g, y):
    ''' computes least square given vector g and y
    '''
    if not isinstance(g, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError('g and y has to be np.ndarrays!')
    square = np.sum((g - y)**2)
    return square
    

def spline_fun(knot_vec, beta, X):
    ''' computes spline value given beta and knot vector
    '''
    val = []
    
    for x in X:
        base_val = basis_N(knot_vec, x)
        val.append(np.dot(beta, base_val))
    return np.array(val)
    
    
def basis_N(knot_vec, X):
    '''
    computes function value Nk of position X given knot vector
    '''
    
    N_vec = [1, X]
    for k, knot in enumerate(knot_vec[0:-2]):
        dk = ((X - knot)**3 - (X - knot_vec[-1])**3) / (knot_vec[-1] - knot)
        dkm = ((X - knot_vec[-2])**3 - (X - knot_vec[-1])**3) / (knot_vec[-1] - knot_vec[-2])
        N_vec.append(dk - dkm)
    return np.array(N_vec)


# sample some points along the test intervall [0,1]
num_p = 25
sp = samplingplan(2)
X = sp.rlh(num_p)
# self.X = sp.optimallhc(num_p)

# 2d case, set all y-value to same value
X[:, 1] = 0.5

# Import and run testfunction Branin/rosenbrock, for one y-value - 2D
tfun = testfun().branin
y = tfun(X)

# 2d case, set all y-value to same value
X = X[:, 0]

# Define number of knots, K st. Equal K unknowns
k = 4
# knot_vec = np.linspace(0, 1, num=k + 2, endpoint=True, retstep=False, dtype=None)[1:-1] # remove first and last values
knot_vec = np.linspace(0, 1, num=k, endpoint=True, retstep=False, dtype=None)

# Plot the basis vectors

xb_plot = np.linspace(0, 1, num=100)
yb_plot = []
fig = plt.figure()
for xv in xb_plot:
    yb_plot.append(basis_N(knot_vec, xv).tolist())

yb_plot = np.array(yb_plot)
for i in np.arange(k):
    plt.plot(xb_plot, yb_plot[:, i])
plt.show()
    
    
    

# Find the optimal beta values, using for example SQP
x0 = np.zeros((k,))
fun = lambda beta: least_square(spline_fun(knot_vec, beta, X), y)
opt_res = scipy.optimize.minimize(fun, x0, method='SLSQP', options={'disp': True, 'maxiter': 1e5})

fig = plt.figure()
###
x_p = np.linspace(0, 1, num=100)
y_p = spline_fun(knot_vec, opt_res.x, x_p)

plt.plot(x_p, y_p)
plt.plot(X, y, 'ro')
plt.show()

# Plot the result and MMSE value
