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
    
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
    
        num_p = 30
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan(2)
        self.RMSE_mean = []
        self.RMSE_std = []
        # self.X = sp.rlh(num_p)
        # self.X = sp.grid(num_p)
        # self.X = sp.MC(num_p)
        self.X = sp.optimallhc(num_p)
        minx, maxx, miny, maxy = [-2, 2, -2, 2]
        # 
        self.X[:, 0] = minx + (maxx - minx) * self.X[:, 0]
        self.X[:, 1] = miny + (maxy - miny) * self.X[:, 1]
        # self.testfun = pyKriging.testfunctions().branin
        self.testfun = pyKriging.testfunctions().rosenbrock
        self.y = self.testfun(self.X)
        
    # # def test_plot_spline_basis_fun(self):
    # #     krig_spline = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='Bspline')
    # #     krig_spline.train()
    # #     krig_spline.Bspl.evaluate()
    # # 
    # #     vis_comp = vis.VisSurfTriangle()
    # #     krig_spline.Bspl.vis = vis_comp
    # #     krig_spline.Bspl.render(colormap=cm.coolwarm)
    # 
    # def test_reg_krig_first(self):
    #     krig_first = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='First')
    #     krig_first.train()
    # 
    #     # And plot the results
    #     krig_first.plot()
    #     # Or the trend function
    #     # krig_first.plot_trend()
    #     # Or the rbfs
    #     # krig_first.plot_rad()
    # # 
    # def test_reg_krig_second(self):
    #     krig_second = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='Second')
    #     krig_second.train()
        # 
        # # And plot the results
        # krig_second.plot()
        # # Or the trend function
        # # krig_second.plot_trend()
        # # krig_second.plot_rad()
    # 
    # # def test_controlPointsOpt(self):
    # #     
    # #     y = testfun(self.X)
    # # 
    # #     krig_spline = regression_kriging(self.X, y, testfunction=testfun, reg='Bspline')
    # #     # krig_spline.train()
    # #     krig_spline.updateData()
    # #     krig_spline.controlPointsOpt(krig_spline.Bspl, self.X, y, np.diag(np.ones((len(self.X),))))
    # # 
    # #     pdb.set_trace()
    # #     krig_spline.Bspl.evaluate()
    # #     krig_spline.Bspl.render()
    # #     pdb.set_trace()
    
    def test_reg_krig_spline(self):
        krig_spline = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='Bspline')
        krig_spline.train()
    
        # And plot the results
        krig_spline.plot()
        # krig_spline.plot_trend()
        # krig_spline.plot_rad()
    


        
        
