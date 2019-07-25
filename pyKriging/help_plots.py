import numpy as np
import matplotlib.pyplot as plt
import pdb
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
    type = ['closed', 'open', 'nonlinear', 'special', 'special2', 'special3']
    # fig = plt.figure()
    # ax = fig.gca()
    color = ['r', 'b', 'k', 'g', 'c', 'm', 'y', 'r', 'b', 'k', 'g', 'c', 'm', 'y', 'r', 'b', 'k', 'g', 'c', 'm', 'y']
    for ind, setup in enumerate(type):

        if setup == 'open':
            Bspl.knotvector_u = tuple(np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist())

            # span_u = helpers.find_span_linear(Bspl.degree_u, Bspl.knotvector_u, ctrlpts_size_u, 1)
            # # span_u = self._span_func(Bspl.degree_u, Bspl.knotvector_u, ctrlpts_size_u, 1)
            # bfunsders_u = helpers.basis_function_ders(Bspl.degree_u, Bspl.knotvector_u, span_u, 1, 0)


        elif setup == 'closed':
            Bspl.knotvector_u = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)

        elif setup == 'nonlinear':
            pdb.set_trace()
            numb = Bspl.degree_u + ctrlpts_size_u + 1
            vec = np.linspace(0.1, 1, num=np.floor(numb / 2)).tolist()

            k = 0.5
            vec = [elem ** k for elem in vec]
            # vec = np.exp(vec)
            # Normalize
            vec = vec / (2 * np.max(vec))

            if numb % 2 == 0:
                # Do not append mid number if uneven number of points!
                lst = np.sort(np.append(vec, -vec)) + 0.5
            else:
                lst = np.sort(np.append(np.append(vec, 0), -vec)) + 0.5
            Bspl.knotvector_u = tuple(lst)

        elif setup == 'special':
            # closed knotvector
            pdb.set_trace()
            c_kn = np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist()

            c_kn[0] = c_kn[1]
            c_kn[-1] = c_kn[-2]

            Bspl.knotvector_u = c_kn

        elif setup == 'special2':
            c_kn = np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist()

            # Closed. nothing fishy
            c_kn[0] = c_kn[2]
            c_kn[1] = c_kn[2]
            c_kn[-1] = c_kn[-3]
            c_kn[-2] = c_kn[-3]
            Bspl.knotvector_u = c_kn

        elif setup == 'special3':
            c_kn = np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist()
            pdb.set_trace()
            # Closed. nothing fishy
            c_kn[0] = c_kn[2]
            c_kn[1] = c_kn[2]
            c_kn[-1] = c_kn[-3]
            c_kn[-2] = c_kn[-3]
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
        fig = plt.figure()
        ax = fig.gca()
        i = 0
        # Bspl.derivatives(1, 1, 0)

        # plot derivative
        bfun_der = []
        basis_p_full = np.array(basis_full(Bspl, basis_p, Bspl.degree_u, spans_p))

        for knot in knots_u:
            span_u = helpers.find_span_linear(Bspl.degree_u, Bspl.knotvector_u, ctrlpts_size_u, knot)
            row = helpers.basis_function_ders(Bspl.degree_u, Bspl.knotvector_u, span_u, knot, 2)[-1]
            # pdb.set_trace()
            bfun_der.append(row)

        der_plot = np.array(basis_full(Bspl, bfun_der, Bspl.degree_u, spans_p))
        # plot the value of the function

        for ind, dummy in enumerate(basis_p_full[0]):
            max_val = np.max(np.abs(der_plot[:, i]))
            ax.plot(knots_u, np.ravel(basis_p_full[:, i]), color[ind])
            ax.plot(knots_u, np.ravel(der_plot[:, i]) * (stop_u - start_u), '--o' + color[ind])  # / np.max(der_plot[:, i])
            i += 1
        ax.set_title(setup)
        plt.show()

    # plt.show()
