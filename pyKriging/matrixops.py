
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
import scipy
import scipy.interpolate as interp
import pdb


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
            
        elif type == 'Bspline':
            # Only for surfaces in this form.
            
            raise NotImplementedError
            
        else:
            print('Unknown trend function type')
            raise ValueError
    
        return f

    def updateData(self):
        
        self.distance = np.zeros((self.n, self.n, self.k))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.distance[i, j] = np.abs((self.X[i]-self.X[j]))  # Every position in this matrix is an array of dimension k!
        
        F = []
        for i in range(0, self.n):
            F.append(self.mean_f(self.X[i], 'First').tolist())
        
        self.F = np.array(F)
        
    def updatePsi(self):
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
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
