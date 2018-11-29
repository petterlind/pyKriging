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


class Test_RSM(unittest.TestCase):
        
    def test_RMSD_fun(self):
        
        num_p = 100
        # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
        sp = samplingplan(2)  
        X = sp.rlh(num_p)
        
        testfun = pyKriging.testfunctions().branin
        y = testfun(X)
        
        
        # Create a BSpline surface instance
        surf = BSpline.Surface()

        # Set degrees
        surf.degree_u = 2
        surf.degree_v = 2
        ctrlpts_u = 3
        ctrlpts_v = 3

        # Set control points
        surf.set_ctrlpts(*exchange.import_txt("ex_surface01.cpt", two_dimensional=True))
        
        i_vec = np.linspace(0, 1, num=ctrlpts_u)
        j_vec = np.linspace(0, 1, num=ctrlpts_v)
        initial_CP = []  # np.zeros((6, 6, 3))
        for i in range(0, len(i_vec)):
            for j in range(0, len(j_vec)):
                initial_CP.append([i_vec[i], j_vec[j], 20])
        surf.set_ctrlpts(initial_CP, ctrlpts_u, ctrlpts_v)
        
        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, ctrlpts_u)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, ctrlpts_v)
        
        # Set evaluation delta
        surf.delta = 0.025

        # Evaluate surface points
        surf.evaluate()

        # Plot the control point grid and the evaluated surface
        vis_comp = vis.VisSurfTriangle()
        surf.vis = vis_comp
        
        k = regression_kriging(X, y, testfunction=testfun, name='simple')
        
        new_s = k.controlPointsOpt(surf, X, y, np.diag(np.ones((len(X),))))
        
        new_s.evaluate()
        vis_comp = vis.VisSurfTriangle()
        # new_s.vis = vis_comp
        # new_s.render(colormap=cm.coolwarm)
    
        
