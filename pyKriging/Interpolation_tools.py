__author__ = 'petterlind'
import numpy as np
from matplotlib import pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import animation
import os
import rbf as rbf_trend
from scipy import interpolate as interp 
from Metamodels import metamodel
import pdb

class regression_other(metamodel):
    def __init__(self, X, y, testfunction=None, reg=None, name='', testPoints=None, normtype='std', **kwargs):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.testfunction = testfunction
        self.name = name
        self.n = self.X.shape[0]
        try:
            self.k = self.X.shape[1]
        except:
            self.k = 1
            self.X = self.X.reshape(-1, 1)
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k) * 2.
        self.Lambda = 0
        self.sigma = 0
        self.normRange = []
        self.ynormRange = []
        self.normtype = normtype
        self.normalizeData()  # normalizes the input data!
        self.reg = reg

    def predict(self, X, norm=True):
        '''
        This function returns the prediction of the model at the real world coordinates of X
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        if norm:
            X = self.normX(X)

        return self.inversenormy(self.predict_normalized(X))

    def predict_normalized(self, X):
        ''' Predicts normalized values'''
        if self.k == 2:
            return self.rbfi(X[0], X[1])
        elif self.k == 3:
            return self.rbfi(X[0], X[1], X[2])
        else:
            return NotImplementedError
            
    def train(self):
        '''
        Fits the model parameter, copy of pykriging method to save programming time..
        '''
        if self.reg == 'RbfG':  # Cubic rbf
            if self.k == 2:
                self.rbfi = interp.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='gaussian')
            elif self.k == 3:
                self.rbfi = interp.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='gaussian')
            else:
                raise NotImplementedError
                
        elif self.reg == 'RbfGL':  # Cubic rbf
            if self.k == 2:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='gaussian', reg='First')
            elif self.k == 3:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='gaussian', reg='First')
            else:
                raise NotImplementedError
                
        elif self.reg == 'RbfGC':  # Cubic rbf
            if self.k == 2:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='gaussian', reg='Cubic')
            elif self.k == 3:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='gaussian', reg='Cubic')
            else:
                raise NotImplementedError

        elif self.reg == 'RbfExpC':
            if self.k == 2:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='cubic', reg='Cubic')
            elif self.k == 3:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='cubic', reg='Cubic')
            else:
                raise NotImplementedError
                
        elif self.reg == 'RbfExpL':
            if self.k == 2:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.y, reg='First', function='cubic')
            elif self.k == 3:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='cubic', reg='First')
            else:
                raise NotImplementedError
        
        elif self.reg == 'RbfExp':
            if self.k == 2:
                self.rbfi = interp.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='cubic')
            elif self.k == 3:
                self.rbfi = interp.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='cubic')
            else:
                raise NotImplementedError
                
        elif self.reg == 'RbfExpCo':
            if self.k == 2:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.y, function='cubic', reg='Constant')
            elif self.k == 3:
                self.rbfi = rbf_trend.Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2], self.y, function='Constant', reg='Constant')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
