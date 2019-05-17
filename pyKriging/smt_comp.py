import unittest
import numpy as np
import pdb
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL as vis
import pyKriging
from pyKriging import matrixops
from pyKriging import samplingplan as sp
from geomdl import utilities
from pyKriging.krige import kriging
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter



xt = np.array([[0.], [1.], [2.], [3], [4.]])
yt = np.array([0., 1., 1.5, 0.5, 1.0])



# krig_cube = kriging(self.X, self.y, testfunction=none, reg='Cubic')
krig_cube = kriging(xt, yt)

krig_cube.train()

# And plot the results
krig_cube.plot1d()
krig_cube.plot_likelihood1d()
# krig_cube.plot_trend()
# krig_cube.plot_rad()
pdb.set_trace()
