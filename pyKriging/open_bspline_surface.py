#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2016-2018
"""

import os
from geomdl import BSpline
from geomdl import exchange
import numpy as np
from geomdl.visualization import VisMPL as vis
from matplotlib import cm
import matplotlib.pyplot as plt
import pyKriging
from geomdl import utilities
import pdb

# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Create a BSpline surface instance
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 3
surf.degree_v = 3

# Set control points
# surf.set_ctrlpts(*exchange.import_txt("ex_surface01.cpt", two_dimensional=True))

testfun = pyKriging.testfunctions().branin
ctrlpts_u = 4
ctrlpts_v = 4
i_vec = np.linspace(0, 1, num=ctrlpts_u)
j_vec = np.linspace(0, 1, num=ctrlpts_v)


initial_CP = []  # np.zeros((6, 6, 3))
for i in range(0, len(i_vec)):
    for j in range(0, len(j_vec)):
        initial_CP.append([i_vec[i], j_vec[j], testfun([i_vec[i], j_vec[j]])])
        
surf.set_ctrlpts(initial_CP, ctrlpts_u, ctrlpts_v)

##
# X, Y = np.meshgrid(i_vec, j_vec)
# zs = np.array([testfun([x, y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# # Plot the surface.
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
# 
# # ax.scatter(self.X[:, 0], self.X[:, 1], self.inversenormy(self.y))
# 
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
# ##


# Set knot vectors

surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, ctrlpts_u)
surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, ctrlpts_v)

# surf.knotvector_u = tuple(np.linspace(0, 1, num=surf.degree_u + ctrlpts_u + 1).tolist())
# surf.knotvector_v = tuple(np.linspace(0, 1, num=surf.degree_v + ctrlpts_v + 1).tolist())

# Set evaluation delta
surf.delta = 0.025

# Evaluate surface points
surf.evaluate()

# Plot the control point grid and the evaluated surface
vis_comp = vis.VisSurfTriangle()
surf.vis = vis_comp
surf.render(colormap=cm.coolwarm)

# Good to have something here to put a breakpoint
pass
