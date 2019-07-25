__author__ = 'chrispaulson'
import numpy as np
import scipy
from scipy.optimize import minimize
from .matrixops import matrixops
import copy
from matplotlib import pyplot as plt
import matplotlib
import pylab
import seaborn as sns

        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.

        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:

            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            G
            QR decomposition of the matrix Ft.
        """
        
        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}
        # Set up R
        MACHINE_EPSILON = np.finfo(np.double).eps
        nugget = 10.*MACHINE_EPSILON
        if self.name == 'MFK':
            if self._lvl != self.nlvl:
                # in the case of multi-fidelity optimization
                # it is very probable that lower-fidelity correlation matrix
                # becomes ill-conditionned 
                nugget = 10.* nugget 
        noise = 0.
        tmp_var = theta
        if self.name == 'MFK':
            if self.options['eval_noise']:
                theta = tmp_var[:-1]
                noise = tmp_var[-1]
    
        r = self.options['corr'](theta, self.D).reshape(-1, 1)
        
        R = np.eye(self.nt) * (1. + nugget + noise)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
        
        # Cholesky decomposition of R
        try:
            self.updatePsi()
        except Exception as err:
            #pass
            # print Exception, err
            raise Exception("bad params")

    def predict(self, X):
        '''
        This function returns the prediction of the model at the real world coordinates of X
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        return self.inversenormy(self.predict_normalized(X))

    def predict_var(self, X):
        '''
        The function returns the model's predicted 'error' at this point in the model.
        :param X: new design variable to evaluate, in physical world units
        :return: Returns the posterior variance (model error prediction)
        '''
        X = copy.deepcopy(X)
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

    def infill_objective_mse(self,candidates, args):
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

    def infill(self, points, method='error', addPoint=True):
        '''
        The function identifies where new points are needed in the model.
        :param points: The number of points to add to the model. Multiple points are added via imputation.
        :param method: Two choices: EI (for expected improvement) or Error (for general error reduction)
        :return: An array of coordinates identified by the infill
        '''
        # We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        initX = np.copy(self.X)
        inity = np.copy(self.y)

        # This array will hold the new values we add
        returnValues = np.zeros([points, self.k], dtype=float)
        for i in range(points):
            rand = Random()
            rand.seed(int(time()))
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            if method=='ei':
                evaluator = self.infill_objective_ei
            else:
                evaluator = self.infill_objective_mse

            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=evaluator,
                                  pop_size=155,
                                  maximize=False,
                                  bounder=ec.Bounder([0] * self.k, [1] * self.k),
                                  max_evaluations=20000,
                                  neighborhood_size=30,
                                  num_inputs=self.k)
            final_pop.sort(reverse=True)
            newpoint = final_pop[0].candidate
            returnValues[i][:] = self.inversenormX(newpoint)
            if addPoint:
                self.addPoint(returnValues[i], self.predict(returnValues[i]), norm=True)

        self.X = np.copy(initX)
        self.y = np.copy(inity)
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

        
        Q, G = linalg.qr(Ft, mode='economic')
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception("F is too ill conditioned. Poor combination "
                                "of regression model and observations.")
        
            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par
        
        # Bouhlel
        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))

            # Let's quickly double check that we're at the optimal value by running a quick local optimizaiton
            lopResults = minimize(self.fittingObjective_local, newValues, method='SLSQP', bounds=locOP_bounds, options={'disp': False})
            newValues = lopResults['x']

            # Finally, set our new theta and pl values and update the model again
            for i in range(self.k):
                self.theta[i] = newValues[i]
            for i in range(self.k):
                self.pl[i] = newValues[i + self.k]
            try:
                self.updateModel()
            except:
                pass
            else:
                break

    def fittingObjective(self,candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.k):
                self.theta[i] = entry[i]
            for i in range(self.k):
                self.pl[i] = entry[i + self.k]
            try:
                self.updateModel()
                self.neglikelihood()
                f = self.NegLnLike
            except Exception as e:
                # print 'Failure in NegLNLike, failing the run'
                # print Exception, e
                f = 10000
            fitness.append(f)
        return fitness

    def fittingObjective_local(self, entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f=10000
        for i in range(self.k):
            self.theta[i] = entry[i]
        for i in range(self.k):
            self.pl[i] = entry[i + self.k]
        try:
            self.updateModel()
            self.neglikelihood()
            f = self.NegLnLike
        except Exception as e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000 
            #  if fail function
        return f

    def plot(self, labels=False, show=True):
        '''
        This function plots 2D and 3D models
        :param labels:
        :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
        :return:
        '''
        
        matplotlib.rcParams['font.family'] = "Times New Roman"
        
        if self.k == 3:
            import mayavi.mlab as mlab

            predictFig = mlab.figure(figure='predict')
            # errorFig = mlab.figure(figure='error')
            if self.testfunction:
                truthFig = mlab.figure(figure='test')
            dx = 1
            pts = 25j
            X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
            scalars = np.zeros(X.shape)
            errscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        # errscalars[i][j][k1] = self.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                        scalars[i][j][k1] = self.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])

            if self.testfunction:
                tfscalars = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        for k1 in range(X.shape[2]):
                            tfplot = tfscalars[i][j][k1] = self.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
                plot.compute_normals = False

            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            plot = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
            plot.compute_normals = False
            # errplt = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
            # errplt.compute_normals = False
            if show:
                mlab.show()

        if self.k == 2:
            
            matplotlib.rcParams['font.family'] = "Times New Roman"
            
            fig = plt.figure(figsize=(8, 6))
            samplePoints = list(zip(*self.X))
            # Create a set of data to plot
            plotgrid = 61
            x = np.linspace(self.normRange[0][0], self.normRange[0][1], num=plotgrid)
            y = np.linspace(self.normRange[1][0], self.normRange[1][1], num=plotgrid)

            # x = np.linspace(0, 1, num=plotgrid)
            # y = np.linspace(0, 1, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results

            zs = np.array([self.predict([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            # Z = (Z*(self.ynormRange[1]-self.ynormRange[0]))+self.ynormRange[0]

            #Calculate errors
            zse = np.array([self.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Ze = zse.reshape(X.shape)

            spx = (self.X[:, 0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            spy = (self.X[:,1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            contour_levels = 25

            ax = fig.add_subplot(222)
            CS = pylab.contourf(X,Y,Ze, contour_levels)
            pylab.colorbar()
            pylab.plot(spx, spy,'ow')

            ax = fig.add_subplot(221)
            if self.testfunction:
                # Setup the truth function
                zt = self.testfunction( np.array(list(zip(np.ravel(X), np.ravel(Y)))) )
                ZT = zt.reshape(X.shape)
                CS = pylab.contour(X,Y,ZT,contour_levels ,colors='k',zorder=2)


            # contour_levels = np.linspace(min(zt), max(zt),50)
            if self.testfunction:
                contour_levels = CS.levels
                delta = np.abs(contour_levels[0]-contour_levels[1])
                contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
                contour_levels = np.append(contour_levels, contour_levels[-1]+delta)

            CS = plt.contourf(X, Y, Z, contour_levels, zorder=1)
            pylab.plot(spx, spy,'ow', zorder=3)
            pylab.colorbar()

            ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
            if show:
                pylab.show()

    def saveFigure(self, name=None):
        '''
        Similar to plot, except that figures are saved to file
        :param name: the file name of the plot image
        '''
        if self.k == 3:
            import mayavi.mlab as mlab

            mlab.options.offscreen = True
            predictFig = mlab.figure(figure='predict')
            mlab.clf(figure='predict')
            errorFig = mlab.figure(figure='error')
            mlab.clf(figure='error')
            if self.testfunction:
                truthFig = mlab.figure(figure='test')
                mlab.clf(figure='test')
            dx = 1
            pts = 75j
            X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
            scalars = np.zeros(X.shape)
            errscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        errscalars[i][j][k1] = self.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                        scalars[i][j][k1] = self.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])

            if self.testfunction:
                tfscalars = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        for k1 in range(X.shape[2]):
                            tfscalars[i][j][k1] = self.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig, compute_normals=False)

            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            pred = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
            pred.compute_normals = False
            errpred = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
            errpred.compute_normals = False
            mlab.savefig('%s_prediction.wrl' % name, figure=predictFig)
            mlab.savefig('%s_error.wrl' % name, figure=errorFig)
            if self.testfunction:
                mlab.savefig('%s_actual.wrl' % name, figure=truthFig)
            mlab.close(all=True)
        if self.k == 2:
            samplePoints = list(zip(*self.X))
            # Create a set of data to plot
            plotgrid = 61
            x = np.linspace(0, 1, num=plotgrid)
            y = np.linspace(0, 1, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results

            zs = np.array([self.predict_normalized([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            Z = (Z * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]

            # Calculate errors
            zse = np.array([self.predicterr_normalized([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
            Ze = zse.reshape(X.shape)

            if self.testfunction:
                # Setup the truth function
                zt = self.testfunction(np.array(
                    list(zip(np.ravel((X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]),
                        np.ravel((Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0])))))
                ZT = zt.reshape(X.shape)

            # Plot real world values
            X = (X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            Y = (Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            spx = (self.X[:, 0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            spy = (self.X[:, 1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]

            return spx, spy, X, Y, Z, Ze
        #     fig = plt.figure(figsize=(8, 6))
        #     # contour_levels = np.linspace(min(zt), max(zt),50)
        #     contour_levels = 15
        #     plt.plot(spx, spy, 'ow')
        #     cs = plt.colorbar()
        #
        #     if self.testfunction:
        #         pass
        #     plt.plot(spx, spy, 'ow')
        #
        #     cs = plt.colorbar()
        #     plt.plot(spx, spy, 'ow')
        #
        #     ax = fig.add_subplot(212, projection='3d')
        #     ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
        #
        #     if self.testfunction:
        #         ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
        # if name:
        #     plt.savefig(name)
        # else:
        #     plt.savefig('pyKrigingResult.png')

    def calcuatemeanMSE(self, p2s=200, points=None):
        '''
        This function calculates the mean MSE metric of the model by evaluating MSE at a number of points.
        :param p2s: Points to Sample, the number of points to sample the mean squared error at. Ignored if the points argument is specified
        :param points: an array of points to sample the model at
        :return: the mean value of MSE and the standard deviation of the MSE points
        '''
        if points is None:
            points = self.sp.rlh(p2s)
        values = np.zeros(len(points))
        for enu, point in enumerate(points):
            values[enu] = self.predict_var(point)
        return np.mean(values), np.std(values)

    def snapshot(self):
        '''
        This function saves a 'snapshot' of the model when the function is called. This allows for a playback of the training process
        '''
        self.history['points'].append(self.n)
        self.history['neglnlike'].append(self.NegLnLike)
        self.history['theta'].append(copy.deepcopy(self.theta))
        self.history['p'].append(copy.deepcopy(self.pl))

        self.history['avgMSE'].append(self.calcuatemeanMSE(points=self.testPoints)[0])

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

    def rsquared(self,actual, observed):
        return np.corrcoef(observed, actual)[0,1] ** 2

    def adjrsquares(self, rsquared, obs):
        return 1-(1-rsquared)*((obs-1)/(obs-self.k))   # adjusted R-square

    def chisquared(self, actual, observed):
        actual = np.array(actual)
        observed = np.array(observed)
        return np.sum( np.abs( np.power( (observed-actual)  ,2)/actual ) )
