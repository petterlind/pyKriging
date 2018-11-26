import unittest
import numpy as np
import pdb
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL as vis
from pyKriging import matrixops

class Test_RSM(unittest.TestCase):
        
    def test_RMSD_fun(self):
        
        # Create a BSpline surface instance
        surf = BSpline.Surface()

        # Set degrees
        surf.degree_u = 3
        surf.degree_v = 3

        # Set control points
        surf.set_ctrlpts(*exchange.import_txt("ex_surface01.cpt", two_dimensional=True))

        # Set knot vectors
        surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

        # Set evaluation delta
        surf.delta = 0.025

        # Evaluate surface points
        surf.evaluate()

        # Import and use Matplotlib's colormaps
        from matplotlib import cm

        # Plot the control point grid and the evaluated surface
        vis_comp = vis.VisSurfTriangle()
        surf.vis = vis_comp
        surf.render(colormap=cm.coolwarm)
        
        
        
        matrixops.controlPointsOpt(surf.knotvector_u, surf.knotvector_v, surf.ctrlpts, u, v, p, np.diag(np.ones((len(surf.knotvector_u),)))))
        
        
        
        
        
        pdb.set_trace()
