#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2016-2018
"""

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
import numpy as np

from geomdl.visualization import VisMPL

# Create a B-Spline curve instance
curve = BSpline.Curve()

# Set up the curve
curve.degree = 3
curve.ctrlpts = exchange.import_txt("ex_curve01.cpt")

# Auto-generate knot vector
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

curve.knotvector = np.linspace(0, 1, curve.degree + len(curve.ctrlpts) + 1)
# Set evaluation delta
curve.delta = 0.01

# Evaluate curve
curve.evaluate()

# Plot the control point polygon and the evaluated curve
vis_comp = VisMPL.VisCurve2D()
curve.vis = vis_comp
curve.render()

# Good to have something here to put a breakpoint
pass
