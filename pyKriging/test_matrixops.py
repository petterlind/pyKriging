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
    
        num_p = 6**2
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan(2)
        self.RMSE_mean = []
        self.RMSE_std = []
        # self.X = sp.rlh(num_p)
        self.X = sp.grid(num_p)
        # self.X = sp.MC(num_p)
    # 
    # def test_reg_krig_first(self):
    #     # Next, we define the problem we would like to solve
    #     testfun = pyKriging.testfunctions().branin
    #     y = testfun(self.X)
    # 
    #     krig_first = regression_kriging(self.X, y, testfunction=testfun, reg='First')
    #     krig_first.train()
    # 
    #     # And plot the results
    #     # krig_first.plot()
    #     # Or the trend function
    #     # krig_first.plot_trend()
    # 
    #     RMSE = krig_first.calcuatemeanMSE()
    #     self.RMSE_mean.append(RMSE[0])
    #     self.RMSE_std.append(RMSE[1])
    # 
    # def test_reg_krig_second(self):
    #     # Next, we define the problem we would like to solve
    #     testfun = pyKriging.testfunctions().branin
    #     y = testfun(self.X)
    # 
    #     krig_second = regression_kriging(self.X, y, testfunction=testfun, reg='Second')
    #     krig_second.train()
    # 
    #     # And plot the results
    #     # krig_second.plot()
    #     # pdb.set_trace()
    #     # Or the trend function
    #     # krig_second.plot_trend()
    #     RMSE = krig_second.calcuatemeanMSE()
    #     self.RMSE_mean.append(RMSE[0])
    #     self.RMSE_std.append(RMSE[1])
    
    # def test_controlPointsOpt(self):
    # 
    # 
    #     testfun = pyKriging.testfunctions().branin
    #     y = testfun(self.X)
    # 
    #     krig_spline = regression_kriging(self.X, y, testfunction=testfun, reg='Bspline')
    #     # krig_spline.train()
    #     krig_spline.updateData()
    #     krig_spline.controlPointsOpt(krig_spline.Bspl, self.X, y, np.diag(np.ones((len(self.X),))))
    # 
    #     pdb.set_trace()
    #     krig_spline.Bspl.evaluate()
    #     krig_spline.Bspl.render()
    #     pdb.set_trace()
        
    def test_reg_krig_spline(self):
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(self.X)
    
        krig_spline = regression_kriging(self.X, y, testfunction=testfun, reg='Bspline')
        krig_spline.train()
        
        pdb.set_trace()
        
        # And plot the results
        krig_spline.plot()
        krig_spline.plot_trend()
    
        pdb.set_trace()
        RMSE = krig_spline.calcuatemeanMSE()
        
        
    # def test_plot_spline_basis_fun(self):
    #     num_p = 100
    #     # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    #     sp = samplingplan()
    #     self.X = sp.grid(num_p)
    #     # self.X = sp.rlh(num_p)
    # 
    # 
    #     # Next, we define the problem we would like to solve
    #     testfun = pyKriging.testfunctions().branin
    #     y = testfun(self.X)
    # 
    #     krig_spline = regression_kriging(self.X, y, testfunction=testfun, reg='Bspline')
    #     krig_spline.train()
    #     pdb.set_trace()
    #     krig_spline.Bspl.evaluate()
    # 
    #     vis_comp = vis.VisSurfTriangle()
    #     krig_spline.Bspl.vis = vis_comp
    #     krig_spline.Bspl.render(colormap=cm.coolwarm)
    # 
    #     # And plot the results
    #     # krig_spline.plot()
        
        
