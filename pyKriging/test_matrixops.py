import unittest
import numpy as np
import pdb
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL as vis
import pyKriging
from pyKriging import matrixops
from pyKriging import samplingplan
from geomdl import utilities
from pyKriging.regressionkrige import regression_kriging
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Test_RSM(unittest.TestCase):
    
    # def test_reg_krig_first(self):
    # 
    #     num_p = 100
    #     # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    #     sp = samplingplan(2)
    #     X = sp.rlh(num_p)
    #     # X = sp.grid(num_p)
    # 
    #     # Next, we define the problem we would like to solve
    #     testfun = pyKriging.testfunctions().branin
    #     y = testfun(X)
    # 
    #     krig_first = regression_kriging(X, y, testfunction=testfun, reg='First')
    #     krig_first.train()
    # 
    #     # And plot the results
    #     krig_first.plot()
    #     # Or the trend function
    #     krig_first.plot_trend()
    # 
    def test_reg_krig_second(self):
        num_p = 5**2
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan()
        # X = sp.grid(num_p)
        # X = sp.rlh(num_p)
        X = sp.MC(num_p)
        
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(X)
    
        krig_second = regression_kriging(X, y, testfunction=testfun, reg='Second')
        krig_second.train()
    
        # And plot the results
        krig_second.plot()
        # Or the trend function
        krig_second.plot_trend()
    
    def test_reg_krig_spline(self):
        num_p = 5**2
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan()
        # X = sp.grid(num_p)
        # X = sp.rlh(num_p)
        X = sp.MC(num_p)
        
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(X)
    
        krig_spline = regression_kriging(X, y, testfunction=testfun, reg='Bspline')
        krig_spline.train()
    
        pdb.set_trace()
        # And plot the results
        krig_spline.plot()
        krig_spline.plot_trend()
        
        # def test_plot_spline_basis_fun(self):
        #     num_p = 100
        #     # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        #     sp = samplingplan()
        #     X = sp.grid(num_p)
        #     # X = sp.rlh(num_p)
        # 
        # 
        #     # Next, we define the problem we would like to solve
        #     testfun = pyKriging.testfunctions().branin
        #     y = testfun(X)
        # 
        #     krig_spline = regression_kriging(X, y, testfunction=testfun, reg='Bspline')
        #     krig_spline.train()
        #     pdb.set_trace()
        #     krig_spline.Bspl.evaluate()
        # 
        #     vis_comp = vis.VisSurfTriangle()
        #     krig_spline.Bspl.vis = vis_comp
        #     krig_spline.Bspl.render(colormap=cm.coolwarm)
            
            # And plot the results
            # krig_spline.plot()
        
        
