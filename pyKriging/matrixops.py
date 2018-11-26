
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
import scipy
import pdb
from pyKriging import trends
from geomdl import BSpline
from geomdl import exchange


class matrixops():

    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n, self.n), dtype=np.float)
        self.psi = np.zeros((self.n, 1))
        self.one = np.ones(self.n)
        self.mu = None
        self.U = None
        self.F = None
        self.f = None
        self.beta = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()
        
    def controlPointsOpt(knots, cps, P, R):
        ''' Function that returns optimal control points values'''
        pass

    def updateData(self):
        
        self.distance = np.zeros((self.n, self.n, self.k))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.distance[i, j] = np.abs((self.X[i] - self.X[j]))  # Every position in this matrix is an array of dimension k!
                
        # Set the trend functions
        self.F, self.f = trends.trend(self.trend_fun, self.X)

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
        self.psi = np.zeros((self.n, 1))
        newPsi = np.exp(-np.sum(self.theta * np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        self.U = np.linalg.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)


    def neglikelihood(self):
        
        # From KK CHOI article
        # Beta
        a = np.linalg.solve(self.U.T, self.y)
        b = np.linalg.solve(self.U, a)
        Mat1 = self.F.dot(b)

        c = np.linalg.solve(self.U.T, self.F)
        d = np.linalg.solve(self.U, c)
        Mat2 = self.F.dot(d)
        
        self.beta = np.linalg.solve(Mat2, Mat1)
        
        self.LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(self.U))))

        a = np.linalg.solve(self.U.T, self.one.T)
        b = np.linalg.solve(self.U, a)
        c = self.one.T.dot(b)
        
        self.mu = self.F * self.beta
        self.SigmaSqr = ((self.y - self.mu).T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, (self.y - self.mu))))) / self.n
        
        self.NegLnLike = 0.5 * self.LnDetPsi + 0.5 * self.n * np.log(self.SigmaSqr)

    def predict_normalized(self, x):
        for i in range(self.n):
            self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
        
        # z = self.y - self.one.dot(self.mu)
        # a = np.linalg.solve(self.U.T, z)
        # b = np.linalg.solve(self.U, a)
        # c = self.psi.T.dot(b)

        # f = self.mu + c
        # return f[0]
        
        z = self.y - self.beta.dot(F(x))
        a = np.linalg.solve(self.U.T, z)
        b = np.linalg.solve(self.U, a)
        c = self.psi.T.dot(b)

        f = self.f * self.beta + c  # small f in K.K CHOI article ?!
        return f[0]
        
    def predicterr_normalized(self, x):
        for i in range(self.n):
            try:
                self.psi[i] = np.exp(-np.sum(self.theta * np.power((np.abs(self.X[i] - x)), self.pl)))
            except Exception as e:
                print(Exception, e)
        try:
            SSqr = self.SigmaSqr*(1 - self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,self.psi))))
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
            SSqr=self.SigmaSqr*(1+self.Lambda-self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,self.psi))))
        except Exception as e:
            print(Exception,e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr,0.5)[0]
