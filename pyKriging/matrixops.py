
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
from geomdl import helpers
import scipy
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pdb
from geomdl.visualization import VisMPL as vis
from geomdl import BSpline
from geomdl import utilities
from matplotlib import cm

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
        
        # WARNING hard coded stuff!
        # ctrlpts_u = 3
        # ctrlpts_v = 3

        CP = []
        for comp in Bspl.ctrlpts:
            for in_comp in comp:
                CP.append(in_comp)
        
        # if dim == 2:
        #     # Assign
        CP[dim::3] = inp
        # else:
        #     CP[12 + dim] = inp[5]
        
        # Group
        x = CP[0::3]
        y = CP[1::3]
        z = CP[2::3]
        # Hard-code initial and end values. INTE BARA PÅ EN PLATS !
        
        # x[0:ctrlpts_u:] = np.zeros((ctrlpts_u,))
        # x[-ctrlpts_u:-1] = np.ones((ctrlpts_u,))
        # y[0::3] = np.zeros((ctrlpts_v,))
        # y[2::3] = np.ones((ctrlpts_v,))
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
            opt_res = scipy.optimize.minimize(fun, x0, method='SLSQP', options={'disp': False})
            if opt_res.success:
                Bspl = self.update_bspl(Bspl, opt_res.x, dim)
                # self.plot_trend()
            else:
                pdb.set_trace()
                print('optimization failed!')
                raise ValueError
        
        return Bspl
        
    def basis_full(self, basis_u, degree_u, spans_u):
        ''' Adds the missing zeros to the base vector 
            has to be called one time per knot vector'''
        
        # rewrite as a loop
        start_u = self.Bspl._knot_vector_u[self.Bspl._degree_u]
        # stop_u = self.Bspl._knot_vector_u[-(self.Bspl._degree_u + 1)]
        
        [start_u_ind] = [i for i, j in enumerate(self.Bspl._knot_vector_u) if j == start_u]
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
        
        if self.reg is None or type == 'constant':
            # n = len(x)
            # f = np.ones((n,))
            raise NotImplementedError
            
        elif self.reg == 'First':
            # 1, x1, x2
            # F = np.array([[1] * n, [x[0]] * n, [x[1]] * n]).T
            f = np.array([1, x[0], x[1], x[0] * x[1]])
            # f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            # f = np.array([1, x[0], x[1], x[0] * x[1]**4, x[0]**4, x[1]**4])
            return f
            
        elif self.reg == 'Second':
            f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            return f
            
        elif self.reg == 'Third':
            raise NotImplementedError
        
        elif self.reg == 'Bspline':
            
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
            
            # Bspl.knotvector_u = utilities.generate_knot_vector(Bspl.degree_u, ctrlpts_u)
            # Bspl.knotvector_v = utilities.generate_knot_vector(Bspl.degree_v, ctrlpts_v)
            
            Bspl.knotvector_u = tuple(np.linspace(0, 1, num=Bspl.degree_u + ctrlpts_size_u + 1).tolist())
            Bspl.knotvector_v = tuple(np.linspace(0, 1, num=Bspl.degree_v + ctrlpts_size_v + 1).tolist())
            
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
        if self.reg != 'Bspline':
            for i in range(0, self.n):
                F.append(self.mean_f(self.X[i], None).tolist())
                
            self.F = np.array(F)
        elif self.reg == 'Bspline':
            # FUTURE, set bspline variables before this function call
            self.F = self.mean_f(self.X, None)
        else:
            print('Unknown regression type, or reg type not set!')
            raise ValueError
            
    def updatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance, self.pl), axis=2))
        self.Psi = np.triu(newPsi, 1)
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n)) + np.multiply(np.mat(eye(self.n)), np.spacing(1))
        self.U = np.linalg.cholesky(self.Psi)
        self.U = self.U.T  # Upper triangular cholesky decomposition.

    def regupdatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta * np.power(self.distance, self.pl), axis=2))
        self.Psi = np.triu(newPsi, 1)
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        
        # print('Remember to remove this!')
        # self.Psi = np.diag(np.ones((len(self.X),)))
        
        
        self.U = np.linalg.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)

    def neglikelihood(self):
        self.LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(self.U))))

        a = np.linalg.solve(self.U.T, self.one.T)
        b = np.linalg.solve(self.U, a)
        c = self.one.T.dot(b)
        d = np.linalg.solve(self.U.T, self.y)
        e = np.linalg.solve(self.U, d)
        
        self.mu=(self.one.T.dot(e))/c
        
        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,(self.y-self.one.dot(self.mu))))))/self.n
        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def regneglikelihood(self):
        self.LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(self.U))))
        # Weighted least square
        if self.reg == 'First' or self.reg == 'Second' or self.reg == 'Third':
                self.beta = np.linalg.solve(np.matmul(self.F.T, np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.F))), (np.matmul(self.F.T, np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.y)))))
                
                # self.SigmaSqr = ((self.y - self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.one.dot(self.mu)))))) / self.n
                self.SigmaSqr = ((self.y - self.F.dot(self.beta)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.F.dot(self.beta)))))) / self.n
                
        elif self.reg == 'Bspline':
            
            # upd_surf = self.controlPointsOpt(self.Bspl, self.X, self.y, np.diag(np.ones((len(self.X),))))  # change to self.Psi!
            
            upd_surf = self.controlPointsOpt(self.Bspl, self.X, self.y, np.array(self.Psi))
            
            self.beta = []
            for pt in upd_surf.ctrlpts:  # Bspline
                self.beta.append(pt[-1])
            
            # self.SigmaSqr = ((self.y - self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.one.dot(self.mu)))))) / self.n
            self.SigmaSqr = ((self.y - self.F.dot(self.beta)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.F.dot(self.beta)))))) / self.n
        else:
            print('Unknown selection in regneglikelihood function')
            raise NotImplementedError
            
        self.NegLnLike = -1. * (-(self.n / 2.) * np.log(self.SigmaSqr) - 0.5 * self.LnDetPsi)
    
    def trend_fun_val(self, x):
        '''
        Made for plotting the trend function value. 
        Rewrite it and use it in predict_normalized for clearear code interpretation!
        '''
        if self.reg != 'Bspline':
            f = self.mean_f(x, None).dot(self.beta)
            
        elif self.reg == 'Bspline':
            f = self.Bspl.evaluate_single(x)[-1]
        return f

    def predict_normalized(self, x):
        for i in range(self.n):
            self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            
        try:
            
            z = self.y - np.dot(self.F, self.beta)
                
        except:
            print('EXCEPT!!! (constant mean value)')
            z = self.y - self.one.dot(self.mu)
        a = np.linalg.solve(self.U.T, z)
        b = np.linalg.solve(self.U, a)
        c = self.psi.T.dot(b)
            
        try:
            if self.reg != 'Bspline':
                f = self.mean_f(x, None).dot(self.beta) + c
                
            elif self.reg == 'Bspline':
                f = self.Bspl.evaluate_single(x)[-1] + c
        except:
            print('EXCEPT!!! (constant mean value)')
            f = self.mu + c
        return f[0]

    def predicterr_normalized(self, x):
        for i in range(self.n):
            try:
                self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            except Exception as e:
                print(Exception, e)
        try:
            SSqr = self.SigmaSqr * (1 - self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.psi))))
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
                self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))
            except Exception as e:
                print(Exception,e)
        try:
            SSqr = self.SigmaSqr * ( 1 + self.Lambda - self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.psi))))
        except Exception as e:
            print(Exception, e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr, 0.5)[0]
