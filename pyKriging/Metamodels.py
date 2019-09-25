__author__ = 'petterlind'
import numpy as np
from matplotlib import pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import animation
import os
from scipy import interpolate as interp 
from pyKriging.samplingplan import samplingplan as sp
import pdb
from itertools import product

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score



class metamodel():
    def __init__(self, X, y, bounds=None, testfunction=None, reg=None, name='', testPoints=None, MLEP=True, normtype='std', Lambda=0.01, PLS=False, **kwargs):
        
        self.X_orig = copy.deepcopy(X)
        self.y_orig = copy.deepcopy(y)
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        
        self.testfunction = testfunction
        self.flag_penal = MLEP
        self.bounds = bounds
        self.name = name
        self.n = self.X.shape[0]


        self.Lambda = 0
        self.sigma = 0

        self.normtype = normtype  #  std if normalized st std is one, else normalized on interval [0, 1]
        self.normRange = []
        self.ynormRange = []
        self.normalizeData()                        # normalizes the input data!
        self.PLS = PLS
        self.pls2 = None
        if self.PLS:
            self.pls2 = PLSRegression(n_components=self.X.shape[1])
            self.pls2.fit(self.X, self.y)
            self.X = self.PLS_trans(self.X)
            
        try:
            self.k = self.X.shape[1]
        except:
            self.k = 1
            self.X = self.X.reshape(-1, 1)
        
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k) * 2.
        self.sp = sp(self.k)
        self.reg = reg
        # self.updateData()
        # self.updateModel()
        self.thetamin = 1
        self.thetamax = 15
        self.pmin = 1.7
        self.pmax = 2.3
        self.pl = np.ones(self.k) * 2
        self.Lambda_min = 0.01 #1e-2
        self.Lambda_max = 0.1
        self.Lambda = Lambda #0.1 #0.03
                    # regression order
        
    def PLS_trans(self, X):
        # The PLS - regression computes a new basis in which the 
        bm = self.pls2.x_rotations_ # full rotation
        Xt = np.linalg.solve(bm, X.T).T
        # Pick out only first two components of this vector
        Xt = Xt[:, :2]
        return Xt 
        
    def PLS_inv_rot(self, X):
        bm = self.pls2.x_rotations_ # full rotation
        Xr = np.dot(bm, X.T).T
        return Xr
    
    def normX(self, X):
        '''    
        :param X: An array of points (self.k long) in physical world units
        :return X: An array normed to our model range of [0,1] for each dimension
        '''
        scalar = False
        if np.isscalar(X[0]):
            X = [X]
            scalar = True

        X_norm = np.ones(np.shape(X)) * np.nan
        for i, row in enumerate(X):  # for every row
            for j, elem in enumerate(row):  # for every element in every row
                if self.normtype == 'std':  # with standard deviation one!
                    X_norm[i, j] = (elem - self.normRange[j][0]) / self.normRange[j][1]
                else:  # in interval [0,1]
                    X_norm[i, j] = (elem - self.normRange[j][0]) / float(self.normRange[j][1] - self.normRange[j][0])

        if scalar:  # unpack
            [X_norm] = X_norm
            return X_norm

        else:
            return X_norm

    def inversenormX(self, X):
        '''
        :param X: An array of points (with self.k elem) in normalized model units
        :return X : An array of real world units
        '''

        scalar = False
        if np.isscalar(X[0]):
            X = [X]
            scalar = True

        X_inv = np.ones(np.shape(X)) * np.nan
        for i, row in enumerate(X):  # for every row
            for j, elem in enumerate(row):  # for every element in every row
                if self.normtype == 'std':
                    X_inv[i, j] = self.normRange[j][0] + elem * self.normRange[j][1]  # x = mu + u*std(X)
                else:
                    X_inv[i, j] = (elem * float(self.normRange[j][1] - self.normRange[j][0])) + self.normRange[j][0]

        if scalar:  # unpack
            [X_inv] = X_inv
            return X_inv
        else:
            return X_inv

    def normy(self, y):
        '''
        :param y: An array of observed values in real-world units
        :return y: A normalized array of model units in the range of [0,1]
        '''
        if self.normtype == 'std':
            return (y - self.ynormRange[0]) / self.ynormRange[1]  # u = (x-mu)/std(X)
        else:
            return (y - self.ynormRange[0]) / (self.ynormRange[1] - self.ynormRange[0])
    def inversenormy(self, y):
        '''
        :param y: A normalized array of model units in the range of [0,1]
        :return: An array of observed values in real-world units
        '''
        if self.normtype == 'std':
            return self.ynormRange[0] + y * self.ynormRange[1]  # x = mu + u * std(X)
        else:
            return (y * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]

    def normalizeData(self):
        '''
        This function is called when the initial data in the model is set.
        We find the max and min of each dimension and norm that axis to a range of [0,1]
        '''
        # lower and upper bound of data.
        for i in range(self.X.shape[1]): # self.k can be smth different if PLS is used!
            if self.normtype == 'std':
                self.normRange.append([np.mean(self.X[:, i]), np.std(self.X[:, i], dtype=np.float64)])
            else:  # determine the intervals
                self.normRange.append([min(self.X[:, i]), max(self.X[:, i])])

        # Normalize data
        self.X = self.normX(self.X)

        if self.normtype == 'std':
            self.ynormRange.append(np.mean(self.y))
            self.ynormRange.append(np.std(self.y, dtype=np.float64))
        else:  # determine the intervals
            self.ynormRange.append(min(self.y))
            self.ynormRange.append(max(self.y))

        for i in range(self.n):
            self.y[i] = self.normy(self.y[i])
            
    def animate():
        if animate:
            def init():
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_zlim([0, 250])
                ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, label='Metamodel')
                ax.scatter(spx, spy, self.inversenormy(self.y), color='k', label='Experiments')
                ax.legend(prop={'size': 20})
                if self.testfunction is not None:
                    ax.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet')
                ax.set_xlabel('$X_1$')
                ax.set_ylabel('$X_2$')
                ax.set_zlabel('$\mathbf{G}(X_1, X_2)$')

                # ax.legend()
                return fig,

            def animate(i):
                ax.view_init(elev=10., azim=i)
                return fig,

            # Animate
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
            # Save
            anim.save(r'C:\Users\pettlind\Dropbox\KTH\PhD\Article2\animate\animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            
        raise NotImplementedError()

    def plot(self, fig=None, ax=None, labels=False, show=True, plot_int=None, animate=False, only_points=False, name=None):
        '''
        This function plots 2D and 3D models
        :param labels:
        :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
        :return:
        https://stackoverflow.com/questions/13316397/matplotlib-animation-no-moviewriters-available
        '''
        
        if self.k > 2:
            pass
            
        if self.k == 2:
            
            if fig is None:
                fig = plt.figure(figsize=(8, 6))

            # samplePoints = list(zip(*self.inversenormX(self.X_orig)))  # lists of list of every coordiante
            # Create a set of data to plot
            plotgrid = 50
            if plot_int is None:
                x = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), num=plotgrid)
                y = np.linspace(min(self.X[:, 1]), max(self.X[:, 1]), num=plotgrid)
            else:  # boundries
                xmin, xmax, ymin, ymax = plot_int
                x = np.linspace(xmin, xmax, num=plotgrid)
                y = np.linspace(ymin, ymax, num=plotgrid)
            X, Y = np.meshgrid(x, y)
            
            if not only_points:  # compute the true values at all points!
                modeldata = np.asarray([np.ravel(X), np.ravel(Y)]).T
                nor = True
                if self.PLS:
                    # modeldata = self.PLS_trans(self.normX(modeldata))
                    nor = False
                    
                zs = np.array([self.predict(data, norm=nor) for data in modeldata])
                Z = zs.reshape(X.shape)  # non-normed
                
            if self.testfunction is not None and self.X_orig.shape[1]==2:
                testdata = np.array(list(zip(np.ravel(X), np.ravel(Y))))
                zt = self.testfunction(self.inversenormX(self.PLS_inv_rot(testdata)))
                ZT = zt.reshape(X.shape)

            if ax is None:
                # ax = fig.add_subplot(111, projection='3d')
                # ax = Axes3D(fig)
                matplotlib.rcParams['font.family'] = "Times New Roman"
                plt.style.use('seaborn-bright')
            # ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')
                fig2 = plt.figure(figsize=(8, 6))
                ax2 = Axes3D(fig2)
                # ax2.set_xlim([0, 1])
                # ax2.set_ylim([0, 1])
                # ax2.set_zlim([0, 250])
                ax2.scatter(self.X[:, 0], self.X[:, 1], self.inversenormy(self.y), color='k', label='Experiments')

                if not only_points:
                    ax2.plot_wireframe(X, Y, Z, rstride=3, cstride=3, label='Metamodel')
                    if self.testfunction is not None and self.X_orig.shape[1]==2:
                        ax2.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet')
                
                ax2.legend(prop={'size': 20})
                ax2.set_xlabel('$X_1$')
                ax2.set_ylabel('$X_2$')
                ax2.set_zlabel('$\mathbf{G}(X_1, X_2)$')
                my_path = os.path.abspath('.')
                plt.savefig(my_path + '/img/' + name + '.png', format='png', dpi=1000)
                
                if show:
                    plt.show()
                
            else:
                pass


            # pylab.title(self.reg)
            # ax.legend(['Approx fun.', 'True fun.'], loc="upper right")
            # ax.legend(['Approx fun.', 'True fun.'], loc="upper right")
            # Now add the legend with some customizations.
            # legend = ax.legend(loc='upper center', shadow=True)
            # legend = ax.legend(loc='upper center', shadow=True)
            
        if self.k == 1:
            if fig is None:
                fig = plt.figure(figsize=(8, 6))

            # Create a set of data to plot
            plotgrid = 50

            if plot_int is None:
                x_vec = np.linspace(self.normRange[0][0], self.normRange[0][1], num=plotgrid)

            else:
                xmin, xmax = plot_int
                x_vec = np.linspace(xmin, xmax, num=plotgrid)

            # Predict based on the optimized results

            y = np.array([self.predict(np.array(x).reshape(1,)) for x in np.ravel(x_vec)])

            plt.plot(x, y, 'ro')

    def pf(self, mu, coe, MC_num, limit):
        '''
        Computes Pf 
        
        Input:
        mu - vector of mean values
        coe - coefficient of determination
        MC_num - number of mc samples
        '''
        
        # Sample points on the surface using MC
        X = sp(k=distr.shape[1]).MC(int(MC_num))
        
        samples = []
        for m, c in zip(mu, coe):
            samples.append(np.random.normal(m, m*c, MC_num))
            
        samples = np.asarray(samples) # check that these are not transposed!
        nor = True
        if self.PLS:
            mtest = self.PLS_trans(self.normX(samples))  # apply dimension reduction to the training data.
            nor = False
            
        f_vec = np.asarray([self.predict(xs, norm=nor) for xs in mtest])

        return sum(f_vec < 0) / float(MC_num)
    
    def RRMSE_R2(self, k, bounds, n=2500):
        '''
        This function calculates the mean relative MSE metric of the model by evaluating MSE at a number of points and the Coefficient of determiniation.
        :param n: Points to Sample, the number of points to sample the mean squared error at. Ignored if the points argument is specified
        :param points: an array of points to sample the model at
        :return: the mean value of MSE and the standard deviation of the MSE points
        '''
        
        inside = 0
        den = 0
        SS_tot = 0
        SS_res = 0
        f_vec = np.zeros((n,))
        y_vec = np.zeros((n,))
        
        nd = n ** (1 / k)
        xi = []
        
        nump = int(np.floor(nd))
        if nump < 3:
            nump = 3
            
        marrays = np.asarray([np.linspace(0,1,nump) for i in range(k)])
        
        # Scale
        for i in range(k):
            marrays[i, :] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * marrays[i, :]
        
        # All points
        mravel = []
        for items in product(*marrays):
            mravel.append(items)
        
        mravel = np.asarray(mravel)
        mtest = copy.deepcopy(mravel)
        
        if self.PLS:
            mtest = self.PLS_trans(self.normX(mravel))  # apply dimension reduction on the training data.
            
        f_vec = np.asarray([self.predict(xs, norm=False) for xs in mtest])
        y_vec = self.testfunction(mravel)
        y_bar = np.sum(y_vec) / n**2

        # https://en.wikipedia.org/wiki/Root-mean-square_deviation
        for f_i, y_i in zip(f_vec, y_vec):
            inside += (f_i - y_i)**2
            den += y_i
            SS_tot += (y_i - y_bar)**2

        # https://www.sciencedirect.com/science/article/pii/S1364032115013258?via%3Dihub
        # https://stats.stackexchange.com/questions/260615/what-is-the-difference-between-rrmse-and-rmsre?rq=1
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        RMSD = np.sqrt(inside / n**2)
        # RRMSE = np.sqrt(inside / n) / den * 100
        R_sq = 1 - inside / SS_tot

        if RMSD < 0:  # or RMSD > 1: #  or R_sq > 1:  # R_sq can be less than zero! - fits data worse than horizontal line.
            raise ValueError('Something of with error estimate!')

        return R_sq, RMSD  # In percentage!
