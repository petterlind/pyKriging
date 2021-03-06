__author__ = 'chrispaulson'
import numpy as np
import scipy
from scipy.optimize import minimize
from .matrixops import matrixops
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pyKriging.samplingplan import samplingplan as sp
import inspyred
from random import Random
from time import time
from inspyred import ec
import math as m
from matplotlib import cm
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pyKriging.Metamodels import metamodel
import pdb
import os
from matplotlib.colors import LightSource

class regression_kriging(metamodel, matrixops):
    
    def __init__(self, X, y, bounds=None, testfunction=None, reg=None, name='', testPoints=None, MLEP=True, normtype='std', Lambda=0.01, PLS=False, **kwargs):
        metamodel.__init__(self, X, y, bounds=bounds, testfunction=testfunction, reg=reg, name=name, testPoints=testPoints, MLEP=MLEP, normtype=normtype, Lambda=Lambda, PLS=PLS, **kwargs)
        matrixops.__init__(self)
        self.updateModel()

    def addPoint(self, newX, newy, norm=True):
        '''
        This add points to the model.
        :param newX: A new design vector point
        :param newy: The new observed value at the point of X
        :param norm: A boolean value. For adding real-world values, this should be True. If doing something in model units, this should be False
        '''
        if norm:
            newX = self.normX(newX)
            newy = self.normy(newy)

        self.X = np.append(self.X, [newX], axis=0)
        self.y = np.append(self.y, newy)
        self.n = self.X.shape[0]

        self.updateData()

        while True:
            try:
                self.updateModel()
            except:
                print('Couldnt update the model with these hyperparameters, retraining')
                self.train()
            else:
                break

    def update(self, values):
        '''
        The function sets new hyperparameters
        :param values: the new theta and p values to set for the model
        '''
        for i in range(self.k):
            self.theta[i] = values[i]
        for i in range(self.k):
            self.pl[i] = values[i + self.k]
        self.Lambda = values[-1]
        self.updateModel()

    def updateModel(self):
        '''
        The function rebuilds the Psi matrix (R) to reflect new data or a change in hyperparamters
        '''
        try:
            self.regupdatePsi()
        except Exception as err:
            # pass
            # print(Exception, err)
            raise Exception("bad params")

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

    def predict_prior(self, X, norm=True):
        '''
        This function returns the prediction of the stochastic process at a coordinate(0)
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        if norm:
            X = self.normX(X)
        return self.inversenormy(self.predict_normalized(X, only_prior=True))

    def predict_var(self, X, norm=True):
        '''
        The function returns the model's predicted 'error' at this point in the model.
        :param X: new design variable to evaluate, in physical world units
        :return: Returns the posterior variance (model error prediction)
        '''
        X = copy.deepcopy(X)
        if norm:
            X = self.normX(X)
        # print X, self.predict_normalized(X), self.inversenormy(self.predict_normalized(X))
        return self.predicterr_normalized(X)

    def expimp(self, x):
        '''
        Returns the expected improvement at the design vector X in the model
        :param x: A real world coordinates design vector
        :return EI: The expected improvement value at the point x in the model
        '''
        S = self.predicterr_normalized(x)
        y_min = np.min(self.y)
        if S <= 0.:
            EI = 0.
        elif S > 0.:
            EI_one = ((y_min - self.predict_normalized(x)) * (0.5 + 0.5*m.erf((
                      1./np.sqrt(2.))*((y_min - self.predict_normalized(x)) /
                                       S))))
            EI_two = ((S * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) *
                      ((y_min - self.predict_normalized(x))**2. / S**2.))))
            EI = EI_one + EI_two
        return EI

    def weightedexpimp(self, x, w):
        """weighted expected improvement (Sobester et al. 2005)"""
        S = self.predicterr_normalized(x)
        y_min = np.min(self.y)
        if S <= 0.:
            EI = 0.
        elif S > 0.:
            EI_one = w*((y_min - self.predict_normalized(x)) * (0.5 +
                        0.5*m.erf((1./np.sqrt(2.))*((y_min -
                                  self.predict_normalized(x)) / S))))
            EI_two = ((1. - w)*(S * (1. / np.sqrt(2. * np.pi))) *
                      (np.exp(-(1./2.) * ((y_min -
                       self.predict_normalized(x))**2. / S**2.))))
            EI = EI_one + EI_two
        return EI

    def sasena_check(self):
        '''' Eq 3.24 from Sasenas PhD thesis, Flexibility and Efficiency Enhancements for Constrained Global Design Optimization with Kriging Approximations

        Checks if MLE(theta, p) < -nlog(VAR(y))

        If true it indicates that the loglikelihood function is monotonic and that the fit might be bad. Altough, assuming OKG, UKG have a more complex behaviour!
        '''
        check = self.SigmaSqr < - self.n * np.log(np.var(self.y))
        if check:
            print('Sasena check true!')
        return check



    def infill_objective_mse(self, candidates, args):
        '''
        This acts
        :param candidates: An array of candidate design vectors from the infill global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated MSE values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.predicterr_normalized(entry))
        return fitness

    def infill_objective_ei(self,candidates, args):
        '''
        The infill objective for a series of candidates from infill global search
        :param candidates: An array of candidate design vectors from the infill global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated Expected Improvement values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.expimp(entry))
        return fitness

    def infill_objective_spacefill(self, candidates, args):
        '''
        The infill objective is optimizing the intersite distance
        :param candidates: An array of candidate design vectors from the infill global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated Expected Improvement values for the candidate population
        '''
        fitness = []

        for entry in candidates:
            min_dist = 100
            for i, point in enumerate(self.X):
                dist = np.linalg.norm(entry - point)
                if min_dist > dist:
                    min_dist = dist
                    # closest = point
                    # index = i

            # Distance is then the objective to minimize, hence minus sign
            fitness.append(-min_dist) # Maximizes this expression
        return fitness

    def infill(self, points, method='spacefill', addPoint=True):
        '''
        The function identifies where new points are needed in the model.
        :param points: The number of points to add to the model. Multiple points are added via imputation.
        :param method: Two choices: EI (for expected improvement) or Error (for general error reduction)
        :return: An array of coordinates identified by the infill
        '''
        # We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        #initX = np.copy(self.X)
        #inity = np.copy(self.y)

        # This array will hold the new values we add
        returnValues = np.zeros([points, self.k], dtype=float)
        for i in range(points):
            rand = Random()
            rand.seed(int(time()))
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology

            if method == 'ei':
                evaluator = self.infill_objective_ei
            elif method == 'spacefill':
                evaluator = self.infill_objective_spacefill
            elif method == 'error':
                evaluator = self.infill_objective_mse
            else:
                raise ValueError('No infill strategy set')

            if self.bounds is None:  # No bounds specified, use interval [0,1] for all data
                Bounds = ec.Bounder([0] * self.k, [1] * self.k)  # full problem, [0,1]
            else:
                Bounds = self.bounds # Note that these can be outside intervall [0.1]

            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=evaluator,
                                  pop_size=155,
                                  maximize=False,
                                  bounder=Bounds,
                                  max_evaluations=20000,
                                  neighborhood_size=30,
                                  num_inputs=self.k)
            final_pop.sort(reverse=True)
            newpoint = final_pop[0].candidate
            returnValues[i][:] = newpoint

            if addPoint:
                self.addPoint(returnValues[i], self.normy(self.predict(returnValues[i], norm=False)), norm=False) # Already normed x-data!

        #self.X = np.copy(initX) # added points and then removed them?!
        #self.y = np.copy(inity)
        self.n = len(self.X)
        self.updateData()
        while True:
            try:
                self.updateModel()
            except:
                self.train()
            else:
                break
        return returnValues

    def generate_population(self, random, args):
        '''
        Generates an initial population for any global optimization that occurs in pyKriging
        :param random: A random seed
        :param args: Args from the optimizer, like population size
        :return chromosome: The new generation for our global optimizer to use
        '''
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        """Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        Optional keyword arguments in args:

        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)

        """
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)

    def train(self, optimizer='ga'):
        '''
        The function trains the hyperparameters of the Kriging model.
        :param optimizer: Two optimizers are implemented, a Particle Swarm Optimizer or a GA
        '''
        # First make sure our data is up-to-date
        self.updateData()
        # Establish the bounds for optimization for theta and p values

        #lowerBound = [self.thetamin] * self.k + [self.pmin] * self.k + [self.Lambda_min]
        #upperBound = [self.thetamax] * self.k + [self.pmax] * self.k + [self.Lambda_max]

        # wo p, lambda
        lowerBound = [self.thetamin]  * self.k
        upperBound = [self.thetamax] * self.k

        #Create a random seed for our optimizer to use
        rand = Random()
        rand.seed(int(time()))
        # If the optimizer option is PSO, run the PSO algorithm
        if optimizer is 'pso':
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=500, # 150
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=10000, # 1000
                                  neighborhood_size=20,
                                  num_inputs=self.k)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)

        # If not using a PSO search, run the GA
        elif optimizer is 'ga':
            ea = inspyred.ec.GA(Random())
            ea.terminator = self.no_improvement_termination
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=50, # 50
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=1000, # 2000, 1000
                                  num_elites=10, # 10
                                  mutation_rate=.05)

        # This code updates the model with the hyperparameters found in the global search
        for entry in final_pop:
            newValues = entry.candidate
            preLOP = copy.deepcopy(newValues)
            locOP_bounds = []
            for i in range(self.k):
                locOP_bounds.append([self.thetamin, self.thetamax])

            #for i in range(self.k):
            #    locOP_bounds.append([self.pmin, self.pmax])

            # locOP_bounds.append([self.Lambda_min, self.Lambda_max])

            # Let's quickly double check that we're at the optimal value by running a quick local optimization
            lopResults = minimize(self.fittingObjective_local, newValues, method='SLSQP', bounds=locOP_bounds, options={'disp': False})
            newValues = lopResults['x']

            # Finally, set our new theta and pl values and update the model again
            for i in range(self.k):
                self.theta[i] = newValues[i]
            #for i in range(self.k):
            #    self.pl[i] = newValues[i + self.k]
            # self.Lambda = newValues[-1]
            try:
                flag_plateau = self.sasena_check()
                self.updateModel()
            except:
                pass
            else:
                break

    def fittingObjective(self, candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            f = 10000
            for i in range(self.k):
                self.theta[i] = entry[i]
            #for i in range(self.k):
            #    self.pl[i] = entry[i + self.k]
            # self.Lambda = entry[-1]
            try:
                self.updateModel()
                self.regneglikelihood()
                f = self.NegLnLike
            except Exception as e:
                #print('Failure in NegLNLike, failing the run')
                # print(Exception, e)
                f = 10000
            fitness.append(f)
        return fitness

    def fittingObjective_local(self, entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f = 10000
        for i in range(self.k):
            self.theta[i] = entry[i]
        #for i in range(self.k):
        #    self.pl[i] = entry[i + self.k]
        # self.Lambda = entry[-1]
        try:
            self.updateModel()
            self.regneglikelihood()
            f = self.NegLnLike
        except Exception as e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000
        return f

    def plot_likelihood(self): 
        ''' Plots the likeliohood function '''
        X, Y = np.meshgrid(np.linspace(self.thetamin, self.thetamax, 200), np.linspace(self.thetamin, self.thetamax, 200))

        Z = []
        self.pl[0] = 2
        self.pl[1] = 2
        tot_l = len(np.ravel(X))
        index = 0
        disp_v = 0
        for x, y in zip(np.ravel(X), np.ravel(Y)):  

            self.theta[0] = x
            self.theta[1] = y

            try:
                self.updateModel()
                self.regneglikelihood()
                f = self.NegLnLike

            except Exception as e:
                #print('Failure in NegLNLike, failing the run')
                # print(Exception, e)
                f = np.nan

            index += 1
            indicator = index / tot_l
            if indicator >= disp_v:
                print('----------------------')
                print(index)
                print('of')
                print(tot_l)
                print('----------------------')
                disp_v += 0.1
            Z.append(f)

        Z = np.array(Z).reshape(X.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        ax.plot_surface(X, Y, Z, cmap='cividis')

        # ax.scatter(self.X[:, 0], self.X[:, 1], self.inversenormy(self.y))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_trend(self, fig=None, ax=None):
        
        matplotlib.rcParams['font.family'] = "Times New Roman"
        X, Y = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05))
        zs = np.array([self.inversenormy(self.trend_fun_val([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        # real function
        X_Yvec = np.stack((np.ravel(X), np.ravel(Y))).T
        X_unscaled = self.inversenormX(X_Yvec)
        z_real = np.array([self.testfunction(np.array([x, y])) for x, y in X_unscaled])
        Z_r = z_real.reshape(X.shape)
        
        if fig is None:
            fig = plt.figure()
            
        ax = fig.gca(projection='3d')
            
        


        # Plot the surface.
        ls = LightSource(270, 45)
        rgb = ls.shade(Z, cmap=cm.jet, vert_exag=0.1, blend_mode='soft')
        ax.plot_surface(X, Y, Z,  alpha=0.5, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
        # ax.plot_wireframe(X, Y, Z_r)

        ax.scatter(self.X[:, 0], self.X[:, 1], self.inversenormy(self.y),facecolor='black', alpha=1)  # points

        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$G(X_1, X_2)$')
        ax.view_init(elev=7, azim=-65,)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        return fig, ax
        # plt.show()

    def plot_rad(self):
        x = y = np.arange(-1.5, 1.5, 0.1)
        X, Y = np.meshgrid(x, y)

        zs = np.array([self.predict([x, y], norm=False) - self.inversenormy(self.trend_fun_val([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        # real function
        X_Yvec = np.stack((np.ravel(X), np.ravel(Y))).T
        X_unscaled = self.inversenormX(X_Yvec)
        z_real = np.array([self.testfunction(np.array([x, y])) for x, y in X_unscaled])
        Z_r = z_real.reshape(X.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        ax.plot_surface(X, Y, Z, cmap='cividis', antialiased=True)
        ax.plot_wireframe(X, Y, Z_r)

        ax.scatter(self.X[:, 0], self.X[:, 1], self.inversenormy(self.y))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def snapshot(self):
        '''
        This function saves a 'snapshot' of the model when the function is called. This allows for a playback of the training process
        '''
        self.history['points'].append(self.n)
        self.history['neglnlike'].append(self.NegLnLike)
        self.history['theta'].append(copy.deepcopy(self.theta))
        self.history['p'].append(copy.deepcopy(self.pl))

        currentPredictions = []
        if self.history['pointData']!=None:
            for pointprim in self.history['pointData']:
                predictedPoint = self.predict(pointprim['point'])
                currentPredictions.append(copy.deepcopy( predictedPoint) )

                pointprim['predicted'].append( predictedPoint )
                pointprim['mse'].append( self.predict_var(pointprim['point']) ) 
                try:
                    pointprim['gradient'] = np.gradient( pointprim['predicted'] )
                except:
                    pass
        if self.history['lastPredictedPoints'] != []:
            self.history['chisquared'].append( self.chisquared(  self.history['lastPredictedPoints'], currentPredictions  ) )
            self.history['rsquared'].append( self.rsquared( self.history['lastPredictedPoints'], currentPredictions ) )
            self.history['adjrsquared'].append( self.adjrsquares( self.history['rsquared'][-1], len( self.history['pointData'] )  ) )
        self.history[ 'lastPredictedPoints' ] = copy.deepcopy(currentPredictions)

    def rsquared(self, actual, observed):
        return np.corrcoef(observed, actual)[0, 1] ** 2

    def adjrsquares(self, rsquared, obs):
        return 1-(1-rsquared)*((obs-1)/(obs-self.k))   # adjusted R-square

    def chisquared(self, actual, observed):
        actual = np.array(actual)
        observed = np.array(observed)
        return np.sum( np.abs( np.power( (observed-actual)  ,2)/actual ) )
