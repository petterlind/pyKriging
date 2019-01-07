import numpy as np
import matplotlib.pyplot as plt
import pdb
from matrixops import matrixops
from geomdl import utilities as geom_util
from geomdl import helpers


def basis_full(Bspl, basis_u, degree_u, spans_u):
    ''' Adds the missing zeros to the base vector
        has to be called one time per knot vector'''
    
    # rewrite as a loop
    start_u = Bspl._knot_vector_u[Bspl._degree_u]
    # stop_u = Bspl._knot_vector_u[-(Bspl._degree_u + 1)]
    
    ind = [i for i, j in enumerate(Bspl._knot_vector_u) if j == start_u]
    start_u_ind = ind[-1]
    # [stop_u] = [i for i, j in enumerate(Bspl._knot_vector_u) if j == stop_u]
    
    base_full = []
    num = int(np.sqrt(len(Bspl.ctrlpts)))
    for i, ub in enumerate(basis_u):
        ind = spans_u[i] - start_u_ind
        base = [0] * num
        base[ind:ind + len(ub)] = ub
        base_full.append(base)
        
    return base_full
    

def plot_base(ctrlpts_size_u, Bspl):
    type = ['open', 'closed', 'special1', 'special2', 'special3', 'special4']
    fig = plt.figure()
    ax = fig.gca()
    color = ['r', 'b', 'k', 'g', 'c', 'm', 'y']
    for ind, setup in enumerate(type):
        
        if setup == 'open':
            Bspl.knotvector_u = tuple(np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist())
            
        elif setup == 'closed':
            Bspl.knotvector_u = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            
        elif setup == 'log':
            numb = Bspl.degree_u + ctrlpts_size_u + 1
            vec = np.logspace(1e-12, 1, num=np.floor(numb / 2))
            if numb % 2 == 0:
                # Do not append mid number if uneven number of points!
                lst = np.sort(np.append(vec, -vec)) / 20 + 0.5
            else:
                lst = np.sort(np.append(np.append(vec, 0.5), -vec)) / 20 + 0.5
            Bspl.knotvector_u = tuple(lst)
            
        elif setup == 'log':
            numb = Bspl.degree_u + ctrlpts_size_u + 1
            vec = np.logspace(1e-12, 1, num=np.floor(numb / 2))
        
            if numb % 2 == 0:
                # Do not append mid number if uneven number of points!
                lst = np.sort(np.append(vec, -vec)) / 20 + 0.5
            else:
                lst = np.sort(np.append(np.append(vec, 0.5), -vec)) / 20 + 0.5
            Bspl.knotvector_u = tuple(lst)
        
        elif setup == 'ilog':
            numb = Bspl.degree_u + ctrlpts_size_u + 1
            vec = np.logspace(1e-12, 1, num=np.floor(numb / 2))
            if numb % 2 == 0:
                # Do not append mid number if uneven number of points!
                lst = np.sort(np.append(vec, -vec + 20)) / 20
            else:
                lst = np.sort(np.append(np.append(vec, 0.5), -vec + 20)) / 20
            Bspl.knotvector_u = tuple(lst)
            
        elif setup == 'special1':
            # closed knotvector
            c_kn = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            c_kn[-3] = (c_kn[-1] + c_kn[-4]) * 0.5
            c_kn[2] = (c_kn[0] + c_kn[3]) * 0.5
            
            Bspl.knotvector_u = c_kn
            
        elif setup == 'special2':
            # closed knotvector
            c_kn = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            c_kn[-3] = 0.6
            c_kn[2] = 0.4
            
            Bspl.knotvector_u = c_kn
            
        elif setup == 'special3':
            # closed knotvector
            c_kn = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            c_kn[-3] = 0.9
            c_kn[2] = 0.1
            
            Bspl.knotvector_u = c_kn
            
        elif setup == 'special4':
            # closed knotvector
            c_kn = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            c_kn[-3] = 0.5
            c_kn[2] = 0.5
            
            Bspl.knotvector_u = c_kn
            
        else:
            raise NotImplementedError
        
        start_u = Bspl._knot_vector_u[Bspl._degree_u]
        stop_u = Bspl._knot_vector_u[-(Bspl._degree_u + 1)]
        
        # Map variables to valid knot space
        knots_u = start_u + (stop_u - start_u) * np.linspace(0, 1, 100)
            
        spans_p = helpers.find_spans(Bspl.degree_u, Bspl.knotvector_u, ctrlpts_size_u, knots_u, Bspl._span_func)
        basis_p = helpers.basis_functions(Bspl.degree_u, Bspl.knotvector_u, spans_p, knots_u)
        
        basis_p_full = np.array(basis_full(Bspl, basis_p, Bspl.degree_u, spans_p))
        
        # Plot the curve
        # fig = plt.figure()
        # ax = fig.gca()
        i = 0
        
        for dummy in basis_p_full[0]:
            
            ax.plot(np.linspace(0, 1, 100), np.ravel(basis_p_full[:, i]), color[ind])
            i += 1
            
    ax.set_title(setup)
    plt.show()
