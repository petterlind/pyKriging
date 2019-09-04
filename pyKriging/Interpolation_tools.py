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
from pyKriging.Metamodels import metamodel
import pdb

class regression_other(metamodel):
    
    def __init__(self, X, y, testfunction=None, reg=None, name='', testPoints=None, normtype='std', **kwargs):
        metamodel.__init__(self, X, y, testfunction=testfunction, reg=reg, name=name, testPoints=testPoints, normtype=normtype, **kwargs)
    
    # def __init__(self, X, y, testfunction=None, reg=None, name='', testPoints=None, normtype='std', **kwargs):
    #     self.X = copy.deepcopy(X)
    #     self.y = copy.deepcopy(y)
    #     self.testfunction = testfunction
    #     self.name = name
    #     self.n = self.X.shape[0]
    #     try:
    #         self.k = self.X.shape[1]
    #     except:
    #         self.k = 1
    #         self.X = self.X.reshape(-1, 1)
    #     self.theta = np.ones(self.k)
    #     self.pl = np.ones(self.k) * 2.
    #     self.Lambda = 0
    #     self.sigma = 0
    #     self.normRange = []
    #     self.ynormRange = []
    #     self.normtype = normtype
    #     self.normalizeData()  # normalizes the input data!
    #     self.reg = reg

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
        return self.rbfi(*X)
            
    def train(self):
        '''
        Fits the model parameter, copy of pykriging method to save programming time..
        '''
        
        if self.reg == 'RbfG':  # Cubic rbf
            self.rbfi = interp.Rbf(*self.X.T, self.y, function='gaussian')
                
        elif self.reg == 'RbfGL':  # Cubic rbf
            self.rbfi = rbf_trend.Rbf(*self.X.T, self.y, function='gaussian', reg='First')
                
        elif self.reg == 'RbfGC':  # Cubic rbf
                self.rbfi = rbf_trend.Rbf(*self.X.T, self.y, function='gaussian', reg='Cubic')
                
        elif self.reg == 'RbfExpC':
            self.rbfi = rbf_trend.Rbf(*self.X.T, self.y, function='cubic', reg='Cubic')
                
        elif self.reg == 'RbfExpL':
                self.rbfi = rbf_trend.Rbf(*self.X.T, self.y, reg='First', function='cubic')
        
        elif self.reg == 'RbfExp':
                self.rbfi = interp.Rbf(*self.X.T, self.y, function='cubic')
                
        elif self.reg == 'RbfExpCo':
            self.rbfi = rbf_trend.Rbf(*self.X.T, self.y, function='cubic', reg='Constant')
            
        else:
            pdb.set_trace()
            raise NotImplementedError
