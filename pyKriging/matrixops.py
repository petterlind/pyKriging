
import numpy as np
from scipy import linalg as la
from numpy.matlib import rand,zeros,ones,empty,eye
from geomdl import helpers
import scipy
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pdb
# import help_plots
from geomdl.visualization import VisMPL as vis
from geomdl import BSpline
from geomdl import utilities as geom_util
from matplotlib import cm
import pyKriging.natural_cubic as nc
# from pyKriging.regressionkrige import regression_kriging


class matrixops():

    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.psi = np.zeros((self.n, 1))
        self.one = np.ones(self.n)
        self.beta = None
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()
        
    def compute_q(self, Bspl, x, y, R, dim):
        '''
        Computes the weighted least square norm in each direction (dim)
        '''
        # Controlpoint vec
        p = []
        for comp in Bspl.ctrlpts:
            for in_comp in comp:
                p.append(in_comp)
        
        Qdim = np.matmul(self.F, p[dim::3])
        fun_val = np.stack((x[:, 0], x[:, 1], y), axis=-1)
        
        norm = np.dot(Qdim - fun_val[:, dim], np.dot(R, Qdim - fun_val[:, dim]))  # generalized least square
        return norm
        
    def update_bspl(self, Bspl, inp, dim):
        
        CP = []
        for comp in Bspl.ctrlpts:
            for in_comp in comp:
                CP.append(in_comp)
        
        CP[dim::3] = inp
        
        # Group
        x = CP[0::3]
        y = CP[1::3]
        z = CP[2::3]
        Bspl.ctrlpts = list(zip(x, y, z))
        return Bspl
    
    def controlPointsOpt(self, Bspl, X, Y, R):
        '''
        TODO Test other algorithms,
        analytical derivative
        Rewrite fun as def for speed.
        '''
        for dim in range(0, 3):
            x0 = []
            for pt in Bspl.ctrlpts:
                x0.append(pt[dim])
                
            fun = lambda x: self.compute_q(self.update_bspl(Bspl, x, dim), X, Y, R, dim)
            opt_res = scipy.optimize.minimize(fun, x0, method='BFGS', options={'disp': False, 'maxiter': 1e5})
            if opt_res.success:
                Bspl = self.update_bspl(Bspl, opt_res.x, dim)
                # self.plot_trend()
            else:
                print('optimization failed!')
                raise ValueError
        
        return Bspl
        
    def basis_full(self, basis_u, degree_u, spans_u):
        ''' Adds the missing zeros to the base vector 
            has to be called one time per knot vector'''
        
        # rewrite as a loop
        start_u = self.Bspl._knot_vector_u[self.Bspl._degree_u]
        # stop_u = self.Bspl._knot_vector_u[-(self.Bspl._degree_u + 1)]
        
        ind = [i for i, j in enumerate(self.Bspl._knot_vector_u) if j == start_u]
        start_u_ind = ind[-1]
        # [stop_u] = [i for i, j in enumerate(self.Bspl._knot_vector_u) if j == stop_u]
        
        base_full = []
        num = int(np.sqrt(len(self.Bspl.ctrlpts)))
        for i, ub in enumerate(basis_u):
            ind = spans_u[i] - start_u_ind
            base = [0] * num
            base[ind:ind + len(ub)] = ub
            base_full.append(base)
            
        return base_full
        
    def mean_f(self, x, Bspl):
        
        # Produce one row in the F matrix
        if self.reg is None or self.reg.lower() == 'constant':
            f = np.array([1])
            return f
            
        elif self.reg.lower() == 'first':
            # 1, x1, x2
            # F = np.array([[1] * n, [x[0]] * n, [x[1]] * n]).T
            f = np.array([1, x[0], x[1], x[0] * x[1]])
            # f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            # f = np.array([1, x[0], x[1], x[0] * x[1]**4, x[0]**4, x[1]**4])
            return f
            
        elif self.reg.lower() == 'second':
            f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            return f
            
        elif self.reg.lower() == 'third':
            raise NotImplementedError
            
        elif self.reg.lower() == 'cubic':
            ''' Natural cubic spline, implementation from ch 5.3 in The elements of statisitical modeling
            '''
            knots = np.linspace(0, 1, num=3)
            f = nc.basis_2d(x, knots)
            return f
            
        elif self.reg.lower() == 'cubic2':
            ''' Natural cubic spline, implementation from ch 5.3 in The elements of statisitical modeling
            '''
            knots = np.linspace(0, 1, num=4)
            f = nc.basis_2d(x, knots)
            return f
            
        elif self.reg.lower() == 'bspline':
            
            Bspl = BSpline.Surface()
            Bspl.delta = 0.025
            
            degree_u = 2
            degree_v = 2
            
            Bspl.degree_u = degree_u
            Bspl.degree_v = degree_v
            
            # Set ctrlpts
            ctrlpts_size_u = 4
            ctrlpts_size_v = 4
            
            i_vec = np.linspace(0, 1, num=ctrlpts_size_u)
            j_vec = np.linspace(0, 1, num=ctrlpts_size_v)
            initial_CP = []  # np.zeros((6, 6, 3))
            mean_inp = np.sum(self.y) / len(self.y)
            for i in range(0, len(i_vec)):
                for j in range(0, len(j_vec)):
                    initial_CP.append([i_vec[i], j_vec[j], mean_inp])
            Bspl.set_ctrlpts(initial_CP, ctrlpts_size_u, ctrlpts_size_v)
            
            ##### KNOT VECTOR ?! ################
            # Bspl.knotvector_u = geom_util.generate_knot_vector(Bspl.degree_u, ctrlpts_size_u)
            # Bspl.knotvector_v = geom_util.generate_knot_vector(Bspl.degree_v, ctrlpts_size_v)
            
            # open
            Bspl.knotvector_u = tuple(np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist())
            Bspl.knotvector_v = tuple(np.linspace(0, 1, num=Bspl.degree_v + ctrlpts_size_v + 1).tolist())
            ################
            numb = Bspl.degree_u + ctrlpts_size_u + 1
            vec = np.linspace(0.1, 1, num=np.floor(numb / 2)).tolist()
            k = 2
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
            Bspl.knotvector_v = tuple(lst)
            ################################################################
            self.Bspl = Bspl
            
            start_u = Bspl._knot_vector_u[Bspl._degree_u]
            stop_u = Bspl._knot_vector_u[-(Bspl._degree_u + 1)]
            
            start_v = Bspl._knot_vector_u[Bspl._degree_v]
            stop_v = Bspl._knot_vector_u[-(Bspl._degree_v + 1)]
        
            # Map variables to valid knot space
            knots_u = start_u + (stop_u - start_u) * x[:, 0]
            knots_v = start_v + (stop_v - start_v) * x[:, 1]
            
            spans_u = helpers.find_spans(degree_u, Bspl.knotvector_u, ctrlpts_size_u, knots_u, Bspl._span_func)
            spans_v = helpers.find_spans(degree_v, Bspl.knotvector_v, ctrlpts_size_v, knots_v, Bspl._span_func)
    
            basis_u = helpers.basis_functions(degree_u, Bspl.knotvector_u, spans_u, knots_u)
            basis_v = helpers.basis_functions(degree_v, Bspl.knotvector_v, spans_v, knots_v)
            
            # Adds the zeros for the missing bases!
            basis_u_full = self.basis_full(basis_u, degree_u, spans_u)
            basis_v_full = self.basis_full(basis_v, degree_v, spans_v)
            
            plot_base = 0
            
            if plot_base:
                ''' Ad hoc programmed helper function that plot the shape functions of the bspline in one direction, for a few different made up knot vectors.'''
                help_plots.plot_base(ctrlpts_size_u, Bspl)
            
            B_row = None
            for i in range(len(basis_u_full)):
                # Following Nils Carlssons master thesis
                A_conc = np.concatenate(np.outer(basis_u_full[i], basis_v_full[i]))
                
                if B_row is None:
                    B_row = A_conc
                else:
                    B_row = np.append(B_row, A_conc)  # one long row of data, Rewrite for speed?
                        
            F = np.reshape(B_row, (-1, len(A_conc)))  # Sort up the control points in one vector
            return F
            
        else:
            print('Unknown trend function type')
            raise ValueError

    def updateData(self):
        
        self.distance = np.zeros((self.n, self.n, self.k))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.distance[i, j] = np.abs((self.X[i] - self.X[j]))  # Every position in this matrix is an array of dimension k!
        
        F = []
        try: # If not regression kriging its lacking the self.reg option in the class!
            if self.reg.lower() == 'constant' or self.reg.lower() == 'first' or self.reg.lower() == 'second' or self.reg.lower() == 'third':
                for i in range(0, self.n):
                    F.append(self.mean_f(self.X[i], None).tolist())
                
                self.F = np.array(F)
                
            elif self.reg == 'Bspline' or self.reg == 'Cubic' or self.reg == 'Cubic2':
                # FUTURE, set bspline variables before this function call
                self.F = self.mean_f(self.X, None)
            else:
                print('Unknown regression type, or reg type not set!')
                raise ValueError
        except:
            pass
            
    def updatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta * np.power(self.distance, self.pl), axis=2))
        self.Psi = np.triu(newPsi, 1)
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n)) + np.multiply(np.mat(eye(self.n)), np.spacing(1))
        self.U = la.cholesky(self.Psi)
        self.U = self.U.T  # Upper triangular cholesky decomposition.

    def regupdatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta * np.power(self.distance, self.pl), axis=2))
        self.Psi = np.triu(newPsi, 1)
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        self.U = la.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)

    def neglikelihood(self):
        self.LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(self.U))))

        a = la.solve(self.U.T, self.one.T)
        b = la.solve(self.U, a)
        c = self.one.T.dot(b)
        d = la.solve(self.U.T, self.y)
        e = la.solve(self.U, d)
        
        self.mu=(self.one.T.dot(e))/c
        
        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(la.solve(self.U, la.solve(self.U.T,(self.y-self.one.dot(self.mu)))))) / self.n
        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5 * self.LnDetPsi)
        
    def regneglikelihood(self):
        self.LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(self.U))))
        # Weighted least square
        if self.reg.lower() == 'constant' or self.reg.lower() == 'first' or self.reg.lower() == 'second' or self.reg.lower() == 'third' or self.reg.lower() == 'cubic' or self.reg.lower() == 'cubic2':
            
                Ft = np.matmul(self.F.T, la.solve_triangular(self.U, la.solve_triangular(self.U.T, self.F), lower=True))
                Yt = (np.matmul(self.F.T, la.solve_triangular(self.U, la.solve_triangular(self.U.T, self.y), lower=True)))
                
                # Check condition number on relevant matrices Ft and R
                sv1, v1 = la.eig(Ft)
                sv2, v2 = la.eigh(self.Psi)
                max_cond = np.max([np.abs(np.max(sv1) / np.min(sv1)), np.abs(np.max(sv2) / np.min(sv2))])
                
                Machine_eps = np.finfo(np.double).eps
                if max_cond > 1 / Machine_eps:
                    print('R or Ft have bad condition! Bad hyperparameters')
                    raise ValueError
                
                self.beta = la.solve(Ft, Yt)
                # self.SigmaSqr = ((self.y - self.one.dot(self.mu)).T.dot(la.solve(self.U, la.solve(self.U.T, (self.y - self.one.dot(self.mu)))))) / self.n
                self.SigmaSqr = ((self.y - self.F.dot(self.beta)).T.dot(la.solve(self.U, la.solve(self.U.T, (self.y - self.F.dot(self.beta)))))) / self.n
                
        elif self.reg.lower() == 'bspline':
        
            # Test without values on the correlation matrix (weights)
            # upd_surf = self.controlPointsOpt(self.Bspl, self.X, self.y, np.diag(np.ones((len(self.X),))))  # change to self.Psi!
        
            # Optimization in order to find the CP values
            upd_surf = self.controlPointsOpt(self.Bspl, self.X, self.y, np.array(self.Psi))
            self.beta = []
            for pt in upd_surf.ctrlpts:  # Bspline
                self.beta.append(pt[-1])
        
            # self.plot_trend()
            # # Exact same as for polynomials (they are indeed polynomials!)
            # self.beta = la.solve(np.matmul(self.F.T, la.solve(self.U, la.solve(self.U.T, self.F))), (np.matmul(self.F.T, la.solve(self.U, la.solve(self.U.T, self.y)))))
            # 
            # self.SigmaSqr = ((self.y - self.one.dot(self.mu)).T.dot(la.solve(self.U, la.solve(self.U.T, (self.y - self.one.dot(self.mu)))))) / self.n
            self.SigmaSqr = ((self.y - self.F.dot(self.beta)).T.dot(la.solve(self.U, la.solve(self.U.T, (self.y - self.F.dot(self.beta)))))) / self.n
        else:
            print('Unknown selection in regneglikelihood function')
            raise NotImplementedError
            
        self.NegLnLike = -1. * (-(self.n / 2.) * np.log(self.SigmaSqr) - 0.5 * self.LnDetPsi)
    
    def trend_fun_val(self, x_vec):
        '''
        Made for plotting the trend function value. 
        Rewrite it and use it in predict_normalized for clearear code interpretation!
        '''
        
        # check if scalar
        f_v = np.array(np.nan)
        if np.isscalar(x_vec[1]):
            x_vec = [x_vec] # So that the first element in list is x itself!
            
        for x in x_vec:
            if self.reg != 'Bspline' or self.reg == 'Cubic' or self.reg == 'Cubic2':
                f = self.mean_f(x, None).dot(self.beta)
                
            elif self.reg == 'Bspline':
                f = self.Bspl.evaluate_single(x)[-1]
                
            f_v = np.append(f_v, f)
        
        return f_v[~np.isnan(f_v)]

    def predict_normalized(self, x_vec):
        
        # check if scalar
        
        f_v = np.array(np.nan)
        if np.isscalar(x_vec[0]):
            x_vec = [x_vec]  # So that the first element in list is x itself!
            
        for x in x_vec:
        
            for i in range(self.n):
                self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            
            try:
                z = self.y - np.dot(self.F, self.beta)
            except:
                print('EXCEPT!!! (constant mean value)')
                pdb.set_trace()
                z = self.y - self.one.dot(self.mu)
                
            # a = la.solve_triangular(self.U.T, z)
            # b = la.solve_triangular(self.U, a, lower=True)
            
            a = la.solve_triangular(self.U, z, lower=True)
            b = la.solve_triangular(self.U.T, a)
            
            c = self.psi.T.dot(b)
            
            try:
                if self.reg != 'Bspline':
                    f = self.mean_f(x, None).dot(self.beta) + c
                elif self.reg == 'Bspline':
                    f = self.Bspl.evaluate_single(x)[-1] + c
            except:
                print('EXCEPT!!! (constant mean value)')
                f = self.mu + c
            f_v = np.append(f_v, f[0])
            f_v[~np.isnan(f_v)]
        # if isscalar(f_v)
        return f_v[~np.isnan(f_v)]

    def predicterr_normalized(self, x):
        for i in range(self.n):
            try:
                self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            except Exception as e:
                print(Exception, e)
        try:
            SSqr = self.SigmaSqr * (1 - self.psi.T.dot(la.solve(self.U, la.solve(self.U.T, self.psi))))
        except Exception as e:
            print(self.U.shape)
            print(self.SigmaSqr.shape)
            print(self.psi.shape)
            print(Exception, e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr, 0.5)[0]

    def regression_predicterr_normalized(self, x):
        for i in range(self.n):
            try:
                self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            except Exception as e:
                print(Exception,e)
        try:
            SSqr = self.SigmaSqr * ( 1 + self.Lambda - self.psi.T.dot(la.solve(self.U, la.solve(self.U.T, self.psi))))
        except Exception as e:
            print(Exception, e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr, 0.5)[0]
