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

class RMSE():
    def __init__(self, *args, **kwargs):

        num_p = 10**2
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan(2)
        self.RMSE_mean = []
        self.RMSE_std = []
        # self.X = sp.rlh(num_p)
        # [3e-5 5e-4 2e-3] num_p = 10**2 rlh
        
        # self.X = sp.grid(num_p)
        # [1e-3 5e-4 4e-3] num_p = 10**2 rlh
        
        self.X = sp.MC(num_p)
        
    
    def reg_krig_first(self):
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(self.X)
    
        krig_first = regression_kriging(self.X, y, testfunction=testfun, reg='First')
        krig_first.train()
    
        # And plot the results
        # krig_first.plot()
        # Or the trend function
        # krig_first.plot_trend()
        
        RMSE = krig_first.calcuatemeanMSE()
        self.RMSE_mean.append(RMSE[0])
        self.RMSE_std.append(RMSE[1])
    
    def reg_krig_second(self):
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(self.X)
    
        krig_second = regression_kriging(self.X, y, testfunction=testfun, reg='Second')
        krig_second.train()
    
        # And plot the results
        # krig_second.plot()
        # pdb.set_trace()
        # Or the trend function
        # krig_second.plot_trend()
        RMSE = krig_second.calcuatemeanMSE()
        self.RMSE_mean.append(RMSE[0])
        self.RMSE_std.append(RMSE[1])
    
    def reg_krig_spline(self):
        # Next, we define the problem we would like to solve
        testfun = pyKriging.testfunctions().branin
        y = testfun(self.X)
    
        krig_spline = regression_kriging(self.X, y, testfunction=testfun, reg='Bspline')
        krig_spline.train()
    
        # And plot the results
        krig_spline.plot()
        krig_spline.plot_trend()
    
        RMSE = krig_spline.calcuatemeanMSE()
        self.RMSE_mean.append(RMSE[0])
        self.RMSE_std.append(RMSE[1])
        
        pdb.set_trace()

run = RMSE()
run.reg_krig_first()
run.reg_krig_second()
run.reg_krig_spline()
print(run.RMSE_mean)
