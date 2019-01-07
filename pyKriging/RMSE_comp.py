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

        num_p = 25
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan(2)
        # self.X = sp.rlh(num_p)
        self.X = sp.optimallhc(num_p)
        self.bounds = [-2, 2, -2, 2]
        minx, maxx, miny, maxy = self.bounds
        self.X[:, 0] = minx + (maxx - minx) * self.X[:, 0]
        self.X[:, 1] = miny + (maxy - miny) * self.X[:, 1]
        
        self.RRMSE = {'First': None, 'Second': None, 'Third': None, 'Spline': None}
        self.R_sq = {'First': None, 'Second': None, 'Third': None, 'Spline': None}
        
        # self.testfun = pyKriging.testfunctions().branin
        self.testfun = pyKriging.testfunctions().rosenbrock
        self.y = self.testfun(self.X)
        
        # [3e-5 5e-4 2e-3] num_p = 10**2 rlh # non-periodic knot span
        
        # self.X = sp.grid(num_p)
        # [1e-3 5e-4 4e-3] num_p = 10**2 rlh

        # self.X = sp.MC(num_p)
        
    def reg_krig_first(self):
        # Next, we define the problem we would like to solve
        krig_first = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='First')
        krig_first.train()
    
        # And plot the results
        # krig_first.plot()
        # Or the trend function
        # krig_first.plot_trend()
        self.R_sq['First'], self.RRMSE['First'] = krig_first.RRMSE_R2(self.bounds)

    def reg_krig_second(self):
        # Next, we define the problem we would like to solve
    
        krig_second = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='Second')
        krig_second.train()
    
        # And plot the results
        krig_second.plot()
        # pdb.set_trace()
        # Or the trend function
        krig_second.plot_trend()
        
        self.R_sq['Second'], self.RRMSE['Second'] = krig_second.RRMSE_R2(self.bounds)
    
    def reg_krig_spline(self):
        # Next, we define the problem we would like to solve
    
        krig_spline = regression_kriging(self.X, self.y, testfunction=self.testfun, reg='Bspline')
        krig_spline.train()
        
        self.R_sq['Spline'], self.RRMSE['Spline'] = krig_spline.RRMSE_R2(self.bounds)
        # And plot the results
        krig_spline.plot()
        krig_spline.plot_trend()
        
        ## PLOT TREND!
        # # Set evaluation delta
        # krig_spline.Bspl.delta = 0.025
        # 
        # # Evaluate surface points
        # krig_spline.Bspl.evaluate()
        # 
        # # Import and use Matplotlib's colormaps
        # 
        # # Plot the control point grid and the evaluated surface
        # vis_comp = vis.VisSurfTriangle()
        # krig_spline.Bspl.vis = vis_comp
        # krig_spline.Bspl.render(colormap=cm.coolwarm)
        ## END PLOT TREND
        
avg = {'First': 0, 'Second': 0, 'Third': 0, 'Spline': 0}
R_avg = {'First': 0, 'Second': 0, 'Third': 0, 'Spline': 0}
numiter = 50

for i in range(numiter):
    run = RMSE()
    run.reg_krig_first()
    run.reg_krig_second()
    run.reg_krig_spline()
    print(run.RRMSE)
    
    for key in run.RRMSE:
        if run.RRMSE[key] is not None:
            avg[key] += run.RRMSE[key]
            R_avg[key] += run.R_sq[key]
            
for key in avg:
    if avg[key] is not 0:
        avg[key] = avg[key] / float(numiter)
        R_avg[key] = R_avg[key] / float(numiter)
    # elif avg[key] is 0:
        # del avg[key]
print(avg)
print(R_avg)
pdb.set_trace()
