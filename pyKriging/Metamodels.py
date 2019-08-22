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
import pdb


class metamodel():

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
        for i in range(self.k):
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

        # if self.bounds is not None:
        #    self.bounds = self.normX(self.bounds)

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

    def plot(self, fig=None, ax=None, labels=False, show=True, plot_int=None, animate=False, only_points=False, name=None):
        '''
        This function plots 2D and 3D models
        :param labels:
        :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
        :return:
        https://stackoverflow.com/questions/13316397/matplotlib-animation-no-moviewriters-available
        '''
        if self.k == 2:

            if fig is None:
                fig = plt.figure(figsize=(8, 6))


            samplePoints = list(zip(*self.inversenormX(self.X))) # lists of list of every coordiante
            # Create a set of data to plot
            plotgrid = 50
            if plot_int is None:
                x = np.linspace(min(samplePoints[0]), max(samplePoints[0]), num=plotgrid)
                y = np.linspace(min(samplePoints[1]), max(samplePoints[1]), num=plotgrid)
            else:  # boundries
                xmin, xmax, ymin, ymax = plot_int
                x = np.linspace(xmin, xmax, num=plotgrid)
                y = np.linspace(ymin, ymax, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results
            if not only_points:  # compute the true values at all points!
                zs = np.array([self.predict([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
                Z = zs.reshape(X.shape)  # non-normed

                #Calculate errors
                # zse = np.array([self.predict_var([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
                # Ze = zse.reshape(X.shape)

            if self.testfunction is not None:
                zt = self.testfunction(np.array(list(zip(np.ravel(X), np.ravel(Y)))))
                ZT = zt.reshape(X.shape)

            if ax is None:
                # ax = fig.add_subplot(111, projection='3d')
                ax = Axes3D(fig)
                matplotlib.rcParams['font.family'] = "Times New Roman"
                plt.style.use('seaborn-bright')
            # ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')


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



                fig2 = plt.figure(figsize=(8, 6))
                ax2 = Axes3D(fig2)
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax2.set_zlim([0, 250])
                ax2.scatter(samplePoints[0], samplePoints[1], self.inversenormy(self.y), color='k', label='Experiments')


                if not only_points:
                    ax2.plot_wireframe(X, Y, Z, rstride=3, cstride=3, label='Metamodel')
                    ax2.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet')
                ax2.legend(prop={'size': 20})
                ax2.set_xlabel('$X_1$')
                ax2.set_ylabel('$X_2$')
                ax2.set_zlabel('$\mathbf{G}(X_1, X_2)$')
                plt.savefig(r'C:\Users\pettlind\Dropbox\KTH\PhD\Article2\animate\figg' + str(self.X.shape[0]) + '.png', format='png', dpi=1000)
            else:

                pass

            fig2 = plt.figure(figsize=(8, 6))
            ax2 = Axes3D(fig2)
            ax2.scatter(samplePoints[0], samplePoints[1], self.inversenormy(self.y), color='k', label='Experiments')
            if not only_points:
                ax2.plot_wireframe(X, Y, Z, rstride=3, cstride=3, label='Metamodel')

                if self.testfunction is not None:
                    ax2.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet', label='testfunction')

            #ax2.legend(prop={'size': 20})
            ax2.set_xlabel('$X_1$')
            ax2.set_ylabel('$X_2$')
            ax2.set_zlabel('$\mathbf{G}(X_1, X_2)$')


            # pylab.title(self.reg)
            # ax.legend(['Approx fun.', 'True fun.'], loc="upper right")
            # ax.legend(['Approx fun.', 'True fun.'], loc="upper right")

            # Now add the legend with some customizations.
            # legend = ax.legend(loc='upper center', shadow=True)
            # legend = ax.legend(loc='upper center', shadow=True)
            if show:
                plt.show()
            else:
                my_path = os.path.abspath('.')
                plt.savefig(my_path + '/img/' + name + '.png', format='png', dpi=1000)

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
            # pylab.colorbar()
            # pylab.plot(spx, spy,'ow')
            # pylab.xlabel('test1')
            # pylab.ylabel('test2')
            # pylab.title(self.reg)


    def RRMSE_R2(self, n=2500):
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
        
        nd = n ** (1 / self.k)
        
        xi = []
        samplePoints = list(zip(*self.inversenormX(self.X)))
        for ind in range(self.k):
            xi.append(np.linspace(min(samplePoints[ind]), max(samplePoints[ind]), num=nd))
        
        # Do grid [x1, x2, x3, x4 ...]
        mgrid = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                          for a in np.meshgrid(*xi)])
        mravel = [np.ravel(X) for X in mgrid]
        
        # Do evaluations
        f_vec = np.array([self.predict(np.asarray(xs)) for xs in zip(*mravel)])
        y_vec = self.testfunction(np.array(list(zip(*mravel))))
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
