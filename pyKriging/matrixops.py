
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
from geomdl import helpers
import scipy
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pdb
from geomdl.visualization import VisMPL as vis

class matrixops():

    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.psi = np.zeros((self.n, 1))
        self.one = np.ones(self.n)
        self.reg = None             # regression order
        self.beta = None
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()
        
    def compute_q(self, Bspl, x, y, R, dim):
        
        p_u = Bspl.degree_u
        p_v = Bspl.degree_v
        U = Bspl.knotvector_u
        V = Bspl.knotvector_v
        n_u = len(U) - p_u - 1  # 6st
        n_v = len(V) - p_v - 1  # 6st
        
        B_row = None
        for i in range(len(x)):
        
            [u, v] = x[i]  # parameterisation ?!
        
            span_u = helpers.find_span_binsearch(p_u, U, n_u, u)
            f_nz_u = helpers.basis_function(p_u, U, span_u, u)
            
            Nu = np.zeros((n_u,))
            Nu[(span_u - p_u): (span_u + 1)] = f_nz_u
            
            span_v = helpers.find_span_binsearch(p_v, V, n_v, v)
            f_nz_v = helpers.basis_function(p_v, U, span_v, v)
            
            Nv = np.zeros((n_v,))
            Nv[(span_v - p_v): (span_v + 1)] = f_nz_v
            
            # Following Nils Carlssons master thesis
            A_conc = np.concatenate(np.outer(Nu, Nv))
            
            if B_row is None:
                B_row = A_conc
            else:
                B_row = np.append(B_row, A_conc)  # one long row of data, Rewrite for speed?
        
        B = np.reshape(B_row, (-1, len(A_conc)))  # Sort up the control points in one vector
        
        # Controlpoint vec
        p = []
        for comp in Bspl.ctrlpts:
            for in_comp in comp:
                p.append(in_comp)
        
        Qdim = np.matmul(B, p[dim::3])
        fun_val = np.stack((x[:, 0], x[:, 1], y), axis=-1)
        
        norm = np.dot(Qdim - fun_val[:, dim], np.dot(R, Qdim - fun_val[:, dim]))  # generalized least square
        return norm
        
        
    def update_bspl(self, Bspl, inp, dim):
        
        # WARNING hard coded stuff!
        ctrlpts_u = 3
        ctrlpts_v = 3

        CP = []
        for comp in Bspl.ctrlpts:
            for in_comp in comp:
                CP.append(in_comp)
        
        if dim == 2:
            # Assign
            CP[dim::3] = inp
        else:
            CP[12 + dim] = inp[5]
        
        
        # Group
        x = CP[0::3]
        y = CP[1::3]
        z = CP[2::3]
        # Hard-code initial and end values. INTE BARA PÅ EN PLATS !
        x[0:ctrlpts_u:] = np.zeros((ctrlpts_u,))
        x[-ctrlpts_u:-1] = np.ones((ctrlpts_u,))
        
        y[0::3] = np.zeros((ctrlpts_v,))
        y[2::3] = np.ones((ctrlpts_v,))
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
            opt_res = scipy.optimize.minimize(fun, x0, method='SLSQP', options={'disp': True})
            if opt_res.success:
                Bspl = self.update_bspl(Bspl, opt_res.x, dim)
                
            else:
                print('optimization failed!')
                raise ValueError
        
        return Bspl
        
    def mean_f(self, x, type):
        if type is None or type == 'constant':
            n = len(X)
            f = np.ones((n,))
        elif type == 'First':
            # 1, x1, x2
            # F = np.array([[1] * n, [x[0]] * n, [x[1]] * n]).T
             f = np.array([1, x[0], x[1]])
            # f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            # f = np.array([1, x[0], x[1], x[0] * x[1]**4, x[0]**4, x[1]**4])
        elif type == 'Second':
            f = np.array([1, x[0], x[1], x[0] * x[1], x[0]**2, x[1]**2])
            
        elif type == 'Third':
            raise NotImplementedError
            
        else:
            print('Unknown trend function type')
            raise ValueError
    
        return f

    def updateData(self):
        
        self.distance = np.zeros((self.n, self.n, self.k))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.distance[i, j] = np.abs((self.X[i] - self.X[j]))  # Every position in this matrix is an array of dimension k!
        
        F = []
        if self.reg != 'Bspline':
            for i in range(0, self.n):
                F.append(self.mean_f(self.X[i], 'First').tolist())
            self.F = np.array(F)
        elif self.reg == 'Bspline':
            self.F
            # Set F to NTN
        else:
            print('Unknown regression type, or reg type not set!')
            raise ValueError
            
    def updatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n))+np.multiply(np.mat(eye(self.n)),np.spacing(1))
        self.U = np.linalg.cholesky(self.Psi)
        self.U = self.U.T  # Upper triangular cholesky decomposition.

    def regupdatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        self.U = np.linalg.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)


    def neglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

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

        # mu = (self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.y))))/self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.one)))
        
        # self.mu=mu
        self.beta = np.linalg.solve(np.matmul(self.F.T, np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.F))), (np.matmul(self.F.T, np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.y)))))

        # self.SigmaSqr = ((self.y - self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.one.dot(self.mu)))))) / self.n
        self.SigmaSqr = ((self.y - self.F.dot(self.beta)).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.F.dot(self.beta)))))) / self.n
        
        self.NegLnLike = -1. * (-(self.n / 2.) * np.log(self.SigmaSqr) - 0.5 * self.LnDetPsi)

    def predict_normalized(self, x):
        for i in range(self.n):
            self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            
        try:
            z = self.y - np.dot(self.F, self.beta)
        except:
            print('EXCEPT!!!')
            z = self.y-self.one.dot(self.mu)
        a = np.linalg.solve(self.U.T, z)
        b = np.linalg.solve(self.U, a)
        c = self.psi.T.dot(b)

        try:
            f = self.mean_f(x, type='First').dot(self.beta) + c
        except:
            f = self.mu + c
            print('EXCEPT!!!')
        return f[0]

    def predicterr_normalized(self, x):
        for i in range(self.n):
            try:
                self.psi[i]=np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            except Exception as e:
                print(Exception, e)
        try:
            SSqr = self.SigmaSqr * (1 - self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.psi))))
        except Exception as e:
            print(self.U.shape)
            print(self.SigmaSqr.shape)
            print(self.psi.shape)
            print(Exception,e)
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
        return np.power(SSqr,0.5)[0]
