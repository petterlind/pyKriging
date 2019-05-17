import unittest
import numpy as np
import pdb
from geomdl import BSpline
from geomdl import exchange
from geomdl.visualization import VisMPL as vis
import pyKriging
from pyKriging import matrixops
from pyKriging import samplingplan as sp
from geomdl import utilities
from pyKriging.regressionkrige import regression_kriging
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
import pickle
import seaborn as sns
import pandas as pd


def plot_RMSE_sea(All_data, models, test_name, nr_exp):
    ''' boxplot the data from RMSE-run'''
    #matplotlib.rcParams['font.family'] = 'cmu serif'
    matplotlib.rcParams['font.family'] = "Times New Roman"
    
    # MYCKET MYCKET MER INFO HÄR FÖR ATT BYGGA OM DICTEN!
    # https://stackoverflow.com/questions/39344167/grouped-boxplot-with-seaborn
    
    # Create a new dict with correct structure for seaborn plots
    
    data = {}
    data['model'] = []
    data['test'] = []
    data['num'] = []
    data['RMSE'] = []
    data['R_sq'] = []
    for test in test_name:
        for model in models:
            for num_p in range(16, 40, 2):
                for iter in range(nr_exp):
                    data['model'].append(model)
                    data['test'].append(test)
                    data['num'].append(num_p)
                    data['R_sq'].append(All_data[test][model][num_p]['R_sq'][iter])
                    data['RMSE'].append(All_data[test][model][num_p]['average'][iter])
    
    df = pd.DataFrame.from_dict(data)
    
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # loop through rows and see which are rosenbrock, branin
    bool_branin = []
    bool_rose = []
    for fun in df.test:
        if fun == 'branin':
            bool_branin.append(True)
            bool_rose.append(False)
        elif fun == 'rosenbrock':
            bool_branin.append(False)
            bool_rose.append(True)

    bool_cub = []
    for mod in df.model:
        if mod == 'Cubic2':
            bool_cub.append(False)
        else:
            bool_cub.append(True)
        

    df_branin = df[np.logical_and(bool_branin, bool_cub)]
    df_rose = df[np.logical_and(bool_rose, bool_cub)]
    
    fig = plt.figure()
    sns.boxplot(x="num", y="RMSE", hue="model", data=df_branin, palette="husl", showfliers=False) # .set_title("RMSE error, Branin")
    sns.set(context='paper', style='white', palette='husl', font ='Times New Roman', font_scale=1, color_codes=True, rc=None)
    L = plt.legend()
    L.get_texts()[0].set_text('OKG - First order')
    L.get_texts()[1].set_text('OKG - Second order')
    L.get_texts()[2].set_text('SKG')
    
    plt.xlabel("Number of data points in each set")
    plt.ylabel("Root mean square error (RMSE)")
    
    fig.set_size_inches(6, 5)
    plt.tight_layout()
    plt.savefig(r'C:\Users\pettlind\Dropbox\Presentationer\SMD2019\Abstract_Template_SMD2019_LaTeX\RMSE.png', format='png', dpi=1000)
    
    fig = plt.figure()
    sns.boxplot(x="num", y="R_sq", hue="model", data=df_branin, palette="husl", showfliers=False)  #.set_title("R_sq, Branin")
    sns.set(context='paper', style='white', palette='husl', font ='Times New Roman', font_scale=1, color_codes=True, rc=None)
    L = plt.legend()
    L.get_texts()[0].set_text('OKG - First order')
    L.get_texts()[1].set_text('OKG - Second order')
    L.get_texts()[2].set_text('SKG')
    
    plt.xlabel("Number of data points in each set")
    plt.ylabel("Coefficient of determination, $R^2$")
    
    fig.set_size_inches(6, 5)
    plt.tight_layout()
    plt.savefig(r'C:\Users\pettlind\Dropbox\Presentationer\SMD2019\Abstract_Template_SMD2019_LaTeX\COD.png', format='png', dpi=1000)
    
    
    # sns.set(style="whitegrid")
=======
    pdb.set_trace()
    
    sns.boxplot(x="num", y="RMSE", hue="model", data=df, palette="PRGn").set_title("RMSE error")
    sns.boxplot(x="num", y="R_sq", hue="model", data=df, palette="PRGn").set_title("R_sq")
    # sns.set(style="whitegrid")
    
=======
    pdb.set_trace()
    
    sns.boxplot(x="num", y="RMSE", hue="model", data=df, palette="PRGn").set_title("RMSE error")
    sns.boxplot(x="num", y="R_sq", hue="model", data=df, palette="PRGn").set_title("R_sq")
    # sns.set(style="whitegrid")
    
>>>>>>> parent of 16ac6f6... BACKUP
    

def plot_RRMSE(All_data):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
<<<<<<< HEAD
>>>>>>> parent of 16ac6f6... BACKUP
    
    for iter1, test in enumerate(All_data):
        fig = plt.figure(iter1)
        ax = fig.gca()
        for iter2, model in enumerate(All_data[test]):
            
            # x and y data
            x = []
            y = []
            
            for num in All_data[test][model]:
                # Maybe check if not empty first?
                x.append(num)
                y.append(All_data[test][model][num]['average'])
    
=======
    pdb.set_trace()
    
    sns.boxplot(x="num", y="RMSE", hue="model", data=df, palette="PRGn").set_title("RMSE error")
    sns.boxplot(x="num", y="R_sq", hue="model", data=df, palette="PRGn").set_title("R_sq")
    # sns.set(style="whitegrid")
    
    

def plot_RRMSE(All_data):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for iter1, test in enumerate(All_data):
        fig = plt.figure(iter1)
        ax = fig.gca()
        for iter2, model in enumerate(All_data[test]):
            
            # x and y data
            x = []
            y = []
            
            for num in All_data[test][model]:
                # Maybe check if not empty first?
                x.append(num)
                y.append(All_data[test][model][num]['average'])
    
>>>>>>> parent of 16ac6f6... BACKUP
=======
    
    for iter1, test in enumerate(All_data):
        fig = plt.figure(iter1)
        ax = fig.gca()
        for iter2, model in enumerate(All_data[test]):
            
            # x and y data
            x = []
            y = []
            
            for num in All_data[test][model]:
                # Maybe check if not empty first?
                x.append(num)
                y.append(All_data[test][model][num]['average'])
    
>>>>>>> parent of 16ac6f6... BACKUP
=======
    pdb.set_trace()
    
    sns.boxplot(x="num", y="RMSE", hue="model", data=df, palette="PRGn").set_title("RMSE error")
    sns.boxplot(x="num", y="R_sq", hue="model", data=df, palette="PRGn").set_title("R_sq")
    # sns.set(style="whitegrid")
    
    

def plot_RRMSE(All_data):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for iter1, test in enumerate(All_data):
        fig = plt.figure(iter1)
        ax = fig.gca()
        for iter2, model in enumerate(All_data[test]):
            
            # x and y data
            x = []
            y = []
            
            for num in All_data[test][model]:
                # Maybe check if not empty first?
                x.append(num)
                y.append(All_data[test][model][num]['average'])
    
>>>>>>> parent of 16ac6f6... BACKUP
=======
    pdb.set_trace()
    
    sns.boxplot(x="num", y="RMSE", hue="model", data=df, palette="PRGn").set_title("RMSE error")
    sns.boxplot(x="num", y="R_sq", hue="model", data=df, palette="PRGn").set_title("R_sq")
    # sns.set(style="whitegrid")
    
    

def plot_RRMSE(All_data):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for iter1, test in enumerate(All_data):
        fig = plt.figure(iter1)
        ax = fig.gca()
        for iter2, model in enumerate(All_data[test]):
            
            # x and y data
            x = []
            y = []
            
            for num in All_data[test][model]:
                # Maybe check if not empty first?
                x.append(num)
                y.append(All_data[test][model][num]['average'])
    
>>>>>>> parent of 16ac6f6... BACKUP
            # Plot the surface
            ax.plot(x, y, colors[iter2] + '-o', label=model)
            ax.legend()
            ax.grid(True)
        
        ax.set_title(test)
        ax.set_xlabel('Experiments')
        ax.set_ylabel('RRMSE')
    plt.show()
    
    
def save_obj(obj, name):
    with open(r'C:\Users\pettlind\Documents\GitHub\pyKriging\pyKriging\obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(r'C:\Users\pettlind\Documents\GitHub\pyKriging\pyKriging\obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
        
def comp_data(numiter, models, test_name, save=True):
    testfuns = {'branin': pyKriging.testfunctions().branin, 'rosenbrock': pyKriging.testfunctions().rosenbrock}

    # Create the data structure
    All_data = {}
    for test in test_name:
        All_data[test] = {}
        for model in models:
            All_data[test][model] = {}
            for num_p in range(16, 40, 2):
                All_data[test][model][num_p] = {}
                
    try:
        # For various number of random experiments,
        for num_p in range(16, 40, 2):
            for test in test_name:  # for all benchmarks
            
                # Create empty lists
                avg = {'First': [], 'Second': [], 'Third': [], 'Spline': [], 'Cubic': [], 'Cubic2': []}
                R_avg = {'First': [], 'Second': [], 'Third': [], 'Spline': [], 'Cubic': [], 'Cubic2': []}
                
                for i in range(numiter):  # for a certain number of runs
                    
                    RRMSE = {'First': None, 'Second': None, 'Third': None, 'Spline': None, 'Cubic': None, 'Cubic2': None}
                    R_sq = {'First': None, 'Second': None, 'Third': None, 'Spline': None, 'Cubic': None, 'Cubic2': None}
                    
                    # X = sp.rlh(num_p)
                    X = sp.samplingplan().optimallhc(num_p)
                    
                    # X = np.append(X, [[0, 0], [1, 1], [0, 1], [1, 0]], axis=0)
                    
                    if test == 'rosenbrock':  # Change boundaries
                        bounds = [-2, 2, -2, 2]
                        minx, maxx, miny, maxy = bounds
                        X[:, 0] = minx + (maxx - minx) * X[:, 0]
                        X[:, 1] = miny + (maxy - miny) * X[:, 1]
                    y = testfuns[test](X)
                    
                    for model in models:
                        krig_mod = regression_kriging(X, y, testfunction=testfuns[test], reg=model)
                        krig_mod.train()
                        
                        # And plot the results
                        # krig_mod.plot()
                        # krig_mod.plot_trend()
                        
                        R_sq[model], RRMSE[model] = krig_mod.RRMSE_R2()
                    
                    # Save result from each run
                    for key in RRMSE:
                        if RRMSE[key] is not None:
                            avg[key].append(RRMSE[key])
                            R_avg[key].append(R_sq[key])
                            
                for key in avg:  # Save the result to the data structure
                    if len(avg[key]) != 0:
                        All_data[test][key][num_p]['average'] = avg[key]
                        All_data[test][key][num_p]['R_sq'] = R_avg[key]
            print(num_p)
        if save:
            save_obj(All_data, 'Data_RRMSE')
        
    except:
        pdb.set_trace()
        

# models = ['First', 'Second', 'Spline', 'Cubic', 'Cubic2']
# models = ['First', 'Second', 'Cubic', 'Cubic2']  # No spline
models = ['Cubic']
# test_name = ['rosenbrock', 'branin']
test_name = ['branin']
# models = ['First']  # No spline
<<<<<<< HEAD
comp_data(20, models, test_name, save=True)
# All_data = load_obj('Data_RRMSE')
plot_RMSE_sea(All_data, models, test_name, 20)
plt.show()
=======
# comp_data(15, models, test_name, save=True)
All_data = load_obj('Data_RRMSE')
plot_RMSE_sea(All_data, models, test_name, 15)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> parent of 16ac6f6... BACKUP
=======
>>>>>>> parent of 16ac6f6... BACKUP
=======
>>>>>>> parent of 16ac6f6... BACKUP
=======
>>>>>>> parent of 16ac6f6... BACKUP
=======
>>>>>>> parent of 16ac6f6... BACKUP
pdb.set_trace()
plot_RRMSE(All_data)

    
#####  25 + 4 corner points #####
# spline, 3ord, open, 10 runs, rosenbrock 0.0803 MSE
# Krig 2nd 10 runs, rosenbrock 0.0910 MSE
# KRIG 1nd, 10 runs, rosenbrock 0.21.

#####  25 + 0 corner points #####
# Second order knot vector distribution, spline, 3ord, 4 ctrlpts, 10 runs, rosenbrock 0.087 MSE
# Krig 2nd 10 runs, rosenbrock 0.081 MSE
