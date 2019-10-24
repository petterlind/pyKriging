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

import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class metamodel():
    def __init__(self, X, y, bounds=None, testfunction=None, reg=None, name='', testPoints=None, MLEP=True, normtype='std', Lambda=0.01, PLS=False, PLS_order=2, **kwargs):
        
        self.X_orig = copy.deepcopy(X)
        self.y_orig = copy.deepcopy(y)
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        
        self.testfunction = testfunction
        self.flag_penal = MLEP
        self.bounds = bounds
        self.name = name
        self.n = self.X.shape[0] # Nr points
        self.k = self.X.shape[1] # nr dimensions
        
        self.non_feasible_mc = None
        self.feasible_mc = None
        self.feasible_y_mc = None
        self.non_feasible_y_mc = None
        self.non_feasible = None
        self.feasible = None
        self.feasible_y = None
        self.non_feasible_y = None
        

        self.Lambda = 0
        self.sigma = 0

        self.normtype = normtype  #  std if normalized st std is one, else normalized on interval [0, 1]
        self.normRange = []
        self.ynormRange = []
        self.normalizeData()                        # normalizes the input data!
        self.PLS = PLS
        self.pls2 = None
        self.PLS_order = PLS_order
        
        if self.PLS_order > self.X_orig.shape[1]:
            print('Higher PLS than dimension of problem')
            raise(ValueError)
            
        
        # lower so that it fits to at least a 3**dim grid!
        if self.n > 3 ** self.PLS_order:
            self.PLS_order = PLS_order
        else:
            self.PLS_order = int(np.floor(np.log(self.n) / np.log(3)))
        
        if self.PLS:
            # Compute all directions, reduction is done in later step!
            self.pls2 = PLSRegression(n_components=self.PLS_order)
            # if self.k == 1:
                # self.pls2 = PLSRegression(n_components=1)
            # elif self.k == 2:
                # self.pls2 = PLSRegression(n_components=2)
            # elif self.k > 2:
                # self.pls2 = PLSRegression(n_components=3)
            # else:
                # raise ValueError
                
            self.pls2.fit(self.X, self.y)
            self.X = self.pls2.transform(self.X)
            # self.X = self.PLS_trans(self.X)
            
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
        try:
            Xt = np.linalg.solve(bm, X.T).T
        except:
            print(traceback.format_exc())
        # Pick out only first two components of this vector
        if np.isscalar(X[0]):
            raise(ValueError)
            
        Xt = Xt[:, : self.PLS_order] # don't work for pointwise data
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

    def plot(self, fig=None, ax=None, labels=False, show=True, animate=False, only_points=False, name=None, PF=False, bounds=None):
        '''
        This function plots 2D and 3D models
        :param labels:
        :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
        :return:
        https://stackoverflow.com/questions/13316397/matplotlib-animation-no-moviewriters-available
        '''
        
        if self.X_orig.shape[1] == 1000: # DESTROYED!
            dim = self.X_orig.shape[1]
            # Multisubplot!
            
            
            def comp(x1, x2, x0, bounds):
                ''' compute variation in only two variables at the time.
                Input:
                x1 - index first variable
                x2 - index second variable
                bounds - bounds for all variable
                x0 - nominal value'''
            
                x = np.linspace(bounds[x1][0], bounds[x1][1], num=20)
                y = np.linspace(bounds[x2][0], bounds[x2][1], num=20)
            
                # Normalize wrong place!
                # for iter, xp, yp in zip(range(0,len(x)),x,y):
                    # x[iter], y[iter] = self.normX(np.array([xp, yp]))
            
                X, Y = np.meshgrid(x, y) 
            
                modeldata = np.asarray([np.ravel(X), np.ravel(Y)]).T # 2d up to here
            
                pos = np.linspace(0, 9, 10)
                bol1 = pos == x1
                bol2 = pos == x2
                # np.logical_or(pos == x1, pos == x2) 
                modeldata_upd = np.ones((modeldata.shape[0], 10)) * np.nan
                test_data = copy.copy(modeldata_upd)
                
                for ii, xa in enumerate(modeldata):
                    temp = copy.copy(x0)
                    temp[bol1] = xa[0]
                    temp[bol2] = xa[1]
                    modeldata_upd[ii] = copy.copy(temp)
                    test_data[ii] = copy.copy(temp)
                
                # prediction  
                # zs =self.predict(self.PLS_trans(self.normX(modeldata_upd)))
                zs = self.predict(self.pls2.transform(self.normX(modeldata_upd)), norm=False)
                Z = zs.reshape(X.shape)  # non-normed
                zt = self.testfunction(test_data)
                ZT = zt.reshape(X.shape)
                return Z, ZT
            
            
            # specs_fix = np.asarray([{'type': 'surface'}]*5*5).reshape(5, 5).tolist()
            # fig = make_subplots(rows=5, cols=5, specs = specs_fix)
            fig = plt.figure()
            fig, axs = plt.subplots(dim-1, dim-1, sharex='col', sharey='row')
            # Plot
            x = np.linspace(0, 1, num=20)
            y = np.linspace(0, 1, num=20)
            X, Y = np.meshgrid(x, y)
            
            bounds = np.asarray(bounds)
            x0 = 0.5 * bounds[:, 0] + 0.5 * bounds[:, 1]
            
            num_mat = np.linspace(0, (dim-1)**2 - 1, (dim-1)**2).reshape(dim-1, dim-1)
            num_v = []
            for i in range(1, dim):
                for j in range(0, i):
                    Z, ZT = comp(i, j, x0, bounds)
                    num_v.append(num_mat[i-1, j])
                    # ax = fig.add_subplot(dim - 1, dim - 1, numb)#, projection='3d')
                    # ax.contourf(X, Y, Z, rstride=3, cstride=3, label='Metamodel')
                    # ax.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet')
                    # contour_levels = 10
                    try:
                        contour_levels = np.linspace(180, 360, 11)
                        CS = axs[i-1, j].contour(X,Y, -Z,contour_levels ,colors='k', linestyles='solid',zorder=2)
                        # Change contour levels so that they match int 180-340!
                        # contour_levels = CS.levels
                        
                        # delta = np.abs(contour_levels[0]-contour_levels[1])
                        # contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
                        # contour_levels = np.append(contour_levels, contour_levels[-1]+delta)
                        
                        CT = axs[i-1, j].contourf(X, Y, -ZT, contour_levels, cmap='cividis', zorder=1)
                        # ax.plot_surface(X, Y, ZT, )
                        axs[i-1, j].axis('off')
                    except:
                        pdb.set_trace()
                    
            # Add colorbar
            # Remove empty subplots
            for i in range(0, num_mat.size):
                if not (i == np.asarray(num_v)).any():
                    axs.flat[i].set_visible(False) # remove these
            
            # axes = fig.get_axes()[0]
            fig.colorbar(CT, ax=axs.flat)
            
            # Set common x and  y labels
            for ax in axs.flat:
                ax.set(xlabel='x-label', ylabel='y-label')
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
                    
            if show:
                plt.show()

            
        elif self.k == 2:
            
            if fig is None:
                fig = plt.figure(figsize=(8, 6))

            # samplePoints = list(zip(*self.inversenormX(self.X_orig)))  # lists of list of every coordiante
            # Create a set of data to plot
            plotgrid = 50
            if bounds is None:
                x = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), num=plotgrid)
                y = np.linspace(min(self.X[:, 1]), max(self.X[:, 1]), num=plotgrid)
                nor = False
                
            else:  # boundries
                x = np.linspace(bounds[0][0], bounds[0][1], num=plotgrid)
                y = np.linspace(bounds[1][0], bounds[1][1], num=plotgrid)
                
                # Normalize 
                for iter, xp, yp in zip(range(0,len(x)),x,y):
                    x[iter], y[iter] = self.normX(np.array([xp, yp]))
                nor = False
                
            X, Y = np.meshgrid(x, y)
            
            if not only_points:  # compute the true values at all points!
                modeldata = np.asarray([np.ravel(X), np.ravel(Y)]).T
                    
                zs = np.array([self.predict(data, norm=nor) for data in modeldata])
                Z = zs.reshape(X.shape)  # non-normed
                
            if self.testfunction is not None and self.X_orig.shape[1]==2:
                testdata = np.array(list(zip(np.ravel(X), np.ravel(Y))))
                
                if self.PLS: # rotate according to PLS if True
                    testdata = self.PLS_inv_rot(testdata)
                
                zt = self.testfunction(self.inversenormX(testdata))
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
                
                if PF:
                    if self.feasible is not None: # 
                        ax2.scatter(self.feasible[:, 0], self.feasible[:, 1], self.feasible_y, color='g', marker="o", label='Feasible model')
                        ax2.scatter(self.non_feasible[:, 0], self.non_feasible[:, 1], self.non_feasible_y, color='r', marker="o", label='Non Feasible model')
                    
                    
                    if self.feasible_mc is not None: # Monte Carlo
                        ax2.scatter(self.feasible_mc[:, 0], self.feasible_mc[:, 1], self.feasible_y_mc, color='g', marker='s', label='Feasible mc')
                        ax2.scatter(self.non_feasible_mc[:, 0], self.non_feasible_mc[:, 1], self.non_feasible_y_mc, color='r', marker='s', label='Non Feasible mc')
                
                if not only_points:
                    ax2.plot_wireframe(X, Y, Z, rstride=3, cstride=3, label='Metamodel')
                    if self.testfunction is not None and self.X_orig.shape[1]==2:
                        ax2.plot_surface(X, Y, ZT, rstride=3, cstride=3, alpha=0.5, cmap='jet')
                        
                ax2.legend(prop={'size': 20})
                ax2.set_xlabel('$X_1$')
                ax2.set_ylabel('$X_2$')
                ax2.set_zlabel('$\mathbf{G}(X_1, X_2)$')
                my_path = os.path.abspath('.')
                plt.savefig(my_path + '\\img\\' + name + '.png', format='png', dpi=1000)
                
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
            
        elif self.k == 1:
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
            
    def pf(self, mu, coe, MC_num, bounds=[], MC=False, PF=False, threshold=0):
        '''
        Computes Pf 
        
        Input:
        mu - vector of mean values
        coe - coefficient of determination
        MC_num - number of mc samples
        MC - Bool, if pure MC is to be done at the surface
        '''
        # Sample points on the surface using MC
        # X = sp(k=self.X_orig.shape[1]).MC(int(MC_num))
        
        samples = []
        for m, c, bound in zip(mu, coe, bounds):
            vec = np.random.normal(m, m*c, int(MC_num))
            
            if len(bound) > 0: # TRUNCATE!
                vec[vec < bound[0]] = bound[0]
                vec[vec > bound[1]] = bound[1]          
            samples.append(vec)
            
        samples = np.asarray(samples).T
        nor = True
        
        if self.PLS:
            mtest = self.pls2.transform(self.normX(samples))  # apply dimension reduction to the training data.
        else:
            mtest = self.normX(samples)
        if PF:
            f_vec = np.asarray([self.predict(xs, norm=False) for xs in mtest]).reshape(mtest.shape[0])
            
            self.feasible = mtest[f_vec > threshold]
            self.non_feasible =  mtest[f_vec < threshold]
            self.feasible_y = f_vec[f_vec > threshold]
            self.non_feasible_y = f_vec[f_vec < threshold]
            
            self.Pf = sum(f_vec < threshold) / float(MC_num)
            
        if MC:
            f_mc = np.asarray(self.testfunction(samples)).reshape(mtest.shape[0])
            self.Mc = sum(f_mc < threshold) / float(MC_num)
            self.feasible_mc = mtest[f_mc > threshold]
            self.non_feasible_mc =  mtest[f_mc < threshold]
            self.feasible_y_mc = f_mc[f_mc > threshold]
            self.non_feasible_y_mc = f_mc[f_mc < threshold]
        
            if np.isnan(f_mc).any(): # Left a sanity check here!
                print('Probably wrong input into aircraft function!')
                raise ValueError()
                
    def RRMSE_R2(self, k, bounds, n=500):
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
        
        # 
        # 
        # 
        # nd = n ** (1 / k)
        # xi = []
        
        # nump = int(np.floor(nd))
        # if nump < 3:
        #     nump = 3
        
        
        # marrays = np.asarray([np.linspace(0,1,nump) for i in range(k)])
        
        # Do instead LHS - with 100*input samples ?
        marrays = sp(k=k).rlh(n)
        mravel = np.ones(marrays.shape) * np.nan
        
        # Scale
        for i in range(k):
            mravel[:, i] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) * marrays[:, i]
        
        # All points
        # mravel = []
        # for items in product(*marrays):
        #     mravel.append(items)
        mtest = copy.deepcopy(mravel)
        
        if self.PLS:
            mtest = self.pls2.transform(self.normX(mravel))  # apply dimension reduction on the training data.
            
        f_vec = np.asarray([self.predict(xs, norm=False) for xs in mtest])
        y_vec = self.testfunction(mravel)
        y_bar = np.sum(y_vec) / n**2
        
        # https://en.wikipedia.org/wiki/Root-mean-square_deviation
        for f_i, y_i in zip(f_vec, y_vec):
            inside += (f_i - y_i)**2
            SS_tot += (y_i - y_bar)**2
            # den += y_i

        # https://www.sciencedirect.com/science/article/pii/S1364032115013258?via%3Dihub
        # https://stats.stackexchange.com/questions/260615/what-is-the-difference-between-rrmse-and-rmsre?rq=1
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        RMSD = np.sqrt(inside / n**2)
        R_sq = 1 - inside / SS_tot
        
        if RMSD < 0:  # or RMSD > 1: #  or R_sq > 1:  # R_sq can be less than zero! - fits data worse than horizontal line.
            raise ValueError('Something of with error estimate!')
            pdb.set_trace()

        return R_sq, RMSD  # In percentage!
