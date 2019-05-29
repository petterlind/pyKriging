__author__ = 'chrispaulson'
import numpy as np
import math as m
import os
import pickle
import pyKriging
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pdb


class samplingplan():
    def __init__(self, k=2, fix_p=None):
        self.samplingplan = []
        self.k = k
        self.x_fix = fix_p
        self.path = os.path.dirname(pyKriging.__file__)
        self.path = self.path + '/sampling_plans/'
    
    def MC(self, n):
        return np.random.rand(n, 2)
        
    def sphere_opt(self, n):
        ''' Sample n random optimal spacefilling samples within a circle of dimension k and radius 1 centered at origo. Also, already picked values on position self.x_fix are also considered
        source: http://6degreesoffreedom.co/circle-random-sampling/, https://en.wikipedia.org/wiki/N-sphere'''
        
        # Assume R = 1, and that the circle is centered around origo
        R = 1
        ml = 1
        
        n = 20
        # print('n=100, hardcoded')
        
        
        # Initial space filling samples between [0, 1]
        
        x_0 = self.optimallhc(n)
        
        # Initial random samples within a circle
        # x_0 = np.ones((n, self.k)) * np.nan
        x_row = np.ones((self.k,)) * np.nan
        
        for iter, row in enumerate(x_0):
            # x_rand = np.random.uniform(size=self.k)
            x_rand = row
            if self.k == 1:
                print('Sphere sampling not implemented for 1d')
                raise ValueError
            
            # Last random is the radius, r
            r = R * x_rand[0] ** (1 / self.k)
            x_rand[1] = x_rand[1] * 2 * np.pi  # to radians
                
            if self.k > 1:
                x_row[0] = r * np.cos(x_rand[1])  # x1
                x_row[1] = r * np.sin(x_rand[1])  # x2
                
            if self.k > 2:
                raise NotImplementedError
            
            x_0[iter, :] = x_row
            
        # fig = plt.figure()
        # plt.plot(x_0[:, 0], x_0[:, 1], 'bx')
            
        def func(x_rav, x_fix):
            ''' computes the minimum distance between all points in the dataset, objective function'''
            
            x_mat = x_rav.reshape(-1, self.k)
            if x_fix is not None:
                x_full = np.append(x_mat, x_fix)
            
            else:
                x_full = x_mat  # No fixed values in first iteration!
                
            n_ro, n_co = x_full.shape
            
            len_obj = n_ro * (n_ro - 1) / 2
            obj_v = np.ones((int(len_obj), )) * np.nan
            
            k = 0
            for i in np.arange(n_ro):
                for j in np.arange(i+1, n_ro):
                        obj_v[k] = np.linalg.norm(x_full[i, :] - x_full[j, :])
                        k += 1
            
            min_obj = np.min(obj_v)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')            
            # ax.plot(x_full[:, 0], x_full[:, 1], 'ro')
            
            assert ~ np.isnan(min_obj)
            return - min_obj**2
        
        # One constraint per point - i.e. inside the circle
        cons = []
        for i in np.arange(n):
            cons.append({'type': 'ineq', 'fun': lambda x: ml - np.linalg.norm(x[self.k * i: self.k * i + self.k])})
        
        # cons = tuple(cons)
        func_lambda = lambda x: func(x, self.x_fix)
        bounds = [(- 1, 1) for i in range(len(np.ravel(x_0)))]
        # res = opt.minimize(func_lambda, np.ravel(x_0), method='SLSQP', constraints=cons, bounds=bounds)
        # pdb.set_trace()
        # res =
        # res.success = True
        # res.x = x
        
        # if res.success:
        # xres = res.x.reshape(-1, self.k)
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d') 
        ax = fig.add_subplot(111)
        # ax.plot(xres[:, 0], xres[:, 1], 'ro', label='Optimized')
        ax.plot(x_0[:, 0], x_0[:, 1], 'bx', label='Initial')
        # plt.legend()
        pdb.set_trace()
        return xres
            
        # else:
            # print('optimization for sampling within hypersphere failed')
            # raise NotImplementedError  # Do smth?
            
    def grid(self, n):
        ''' generates ordered sampling points, grid-like
        '''
        # self.k is the dimension of the problem
        n_side = np.sqrt(n)
        x_side = np.linspace(0, 1, n_side)
        x1, x2 = np.meshgrid(x_side, x_side)
        
        return np.stack((np.concatenate(x1), np.concatenate(x2)), axis=-1)
        
    def circle(self, n):
        '''https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
        Sample uniformly whithin a circle
        '''
        
        # Cylindrical coord
        R = 0.5
        r = R * np.sqrt(np.random.rand(n, 1))
        theta = np.random.rand(n, 1) * 2 * np.pi
        
        # Cartesian coordinates
        centerX1 = 0.5
        centerX2 = 0.5
        x1 = centerX1 + r * np.cos(theta)
        x2 = centerX2 + r * np.sin(theta)

        return np.stack((np.concatenate(x1), np.concatenate(x2)), axis=-1)
    
    def rlh(self, n, Edges=0):
        """
        Generates a random latin hypercube within the [0,1]^k hypercube

        Inputs:
            n-desired number of points
            k-number of design variables (dimensions)
            Edges-if Edges=1 the extreme bins will have their centers on the edges of the domain

        Outputs:
            Latin hypercube sampling plan of n points in k dimensions
         """

        #pre-allocate memory
        X = np.zeros((n, self.k))

        #exclude 0

        for i in range(0,self.k):
            X[:, i] = np.transpose(np.random.permutation(np.arange(1,n+1,1)))

        if Edges == 1:
            X = (X-1)/(n-1)
        else:
            X = (X-0.5)/n

        return X

    def optimallhc(self, n, population=30, iterations=30, generation=False):
            """
            Generates an optimized Latin hypercube by optimizing the Morris-Mitchell
            criterion for a range of exponents and plots the first two dimensions of
            the current hypercube throughout the optimization process.

            Inputs:
                n - number of points required
                Population - number of individuals in the evolutionary operation
                             optimizer
                Iterations - number of generations the evolutionary operation
                             optimizer is run for
                             
                Note: high values for the two inputs above will ensure high quality
                hypercubes, but the search will take longer.
                
                Minpoints  - Minimum number of points in optimization loop when fixed x-values are used
                
                generation - if set to True, the LHC will be generated. If 'False,' the algorithm will check for an existing plan before generating.

            Output:
                X - optimized Latin hypercube


            """
            ## TODO: This code isnt working in the Python3 branch.

            # if not generation:

                # Check for existing LHC sampling plans
                # if os.path.isfile('{0}lhc_{1}_{2}.pkl'.format(self.path,self.k, n)):
                #     X = pickle.load(open('{0}lhc_{1}_{2}.pkl'.format(self.path,self.k, n), 'rb'))
                #     return X
                # else:
                #     print(self.path)
                #     print('SP not found on disk, generating it now.')

            #list of qs to optimise Phi_q for
            q = [1,2,5,10,20,50,100]
            # q = [2,5,10]

            #Set the distance norm to rectangular for a faster search. This can be
            #changed to p=2 if the Euclidean norm is required.
            p = 2

            #we start with a random Latin hypercube
            XStart = self.rlh(n)
            
            X3D = np.zeros((n,self.k,len(q)))
            #for each q optimize Phi_q
            for i in range(len(q)):
                # print(('Now_optimizing_for_q = %d \n' %q[i]))
                X3D[:,:,i] = self.mmlhs(XStart, population, iterations, q[i])

            #sort according to the Morris-Mitchell criterion
            Index = self.mmsort(X3D, p)
            # print(('Best_lh_found_using_q = %d \n' %q[Index[1]]))

            #and the Latin hypercube with the best space-filling properties is

            X = X3D[:,:,Index[1]]
            # pickle.dump(X, open('{0}lhc_{1}_{2}.pkl'.format(self.path,self.k, n), 'wb'))
            return X


    def fullfactorial(self, ppd=5):
        ix = (slice(0, 1, ppd*1j),) * self.k
        a = np.mgrid[ix].reshape(self.k, ppd**self.k).T
        return a

    def mmsort(self,X3D,p=1):
        """
        Ranks sampling plans according to the Morris-Mitchell criterion definition.
        Note: similar to phisort, which uses the numerical quality criterion Phiq
        as a basis for the ranking.

        Inputs:
            X3D - three-dimensional array containing the sampling plans to be ranked.
            p - the distance metric to be used (p=1 rectangular - default, p=2 Euclidean)

        Output:
            Index - index array containing the ranking

        """
        #Pre-allocate memory
        Index = np.arange(np.size(X3D,axis=2))

        #Bubble-sort
        swap_flag = 1

        while swap_flag == 1:
            swap_flag = 0
            i = 1
            while i<=len(Index)-2:
                if self.mm(X3D[:,:,Index[i]],X3D[:,:,Index[i+1]],p) == 2:
                    arrbuffer=Index[i]
                    Index[i] = Index[i+1]
                    Index[i+1] = arrbuffer
                    swap_flag=1
                i = i + 1
            return Index


    def perturb(self, X, PertNum):
        """
        Interchanges pairs of randomly chosen elements within randomly
        chosen columns of a sampling plan a number of times. If the plan is
        a Latin hypercube, the result of this operation will also be a Latin
        hypercube.

        Inputs:
            X - sampling plan
            PertNum - the number of changes (perturbations) to be made to X.
        Output:
            X - perturbed sampling plan

        """
        X_pert = X.copy()
        [n,k] = np.shape(X_pert)

        for pert_count in range(0,PertNum):
            col = int(m.floor(np.random.rand(1)*k))

            #Choosing two distinct random points
            el1 = 0
            el2 = 0
            while el1 == el2:
                el1 = int(m.floor(np.random.rand(1)*n))
                el2 = int(m.floor(np.random.rand(1)*n))

            #swap the two chosen elements
            arrbuffer = X_pert[el1,col]
            X_pert[el1,col] = X_pert[el2,col]
            X_pert[el2,col] = arrbuffer

        return X_pert

    def mmlhs(self, X_start, population,iterations, q):
        """
        Evolutionary operation search for the most space filling Latin hypercube
        of a certain size and dimensionality. There is no need to call this
        directly - use bestlh.m

        """
        X_s = X_start.copy()

        n = np.size(X_s, 0)

        X_best = X_s

        Phi_best = self.mmphi(X_best)

        leveloff = m.floor(0.85 * iterations)

        for it in range(0,iterations):
            if it < leveloff:
                mutations = int(round(1+(0.5*n-1)*(leveloff-it)/(leveloff-1)))
            else:
                mutations = 1

            X_improved = X_best
            Phi_improved = Phi_best

            for offspring in range(0, population):
                X_try = self.perturb(X_best, mutations)
                Phi_try = self.mmphi(X_try, q)

                if Phi_try < Phi_improved:
                    X_improved = X_try
                    Phi_improved = Phi_try

            if Phi_improved < Phi_best:
                X_best = X_improved
                Phi_best = Phi_improved

        return X_best

    def mmphi(self, X, q=2, p=1):

        """
        Calculates the sampling plan quality criterion of Morris and Mitchell

        Inputs:
            X - Sampling plan
            q - exponent used in the calculation of the metric (default = 2)
            p - the distance metric to be used (p=1 rectangular - default , p=2 Euclidean)

        Output:
            Phiq - sampling plan 'space-fillingness' metric
        """
        #calculate the distances between all pairs of
        #points (using the p-norm) and build multiplicity array J
        
        # Add the fixed points for the evaluation of the quality
        if self.x_fix is not None:
            X = np.append(X, self.x_fix, axis=0)
        
        J, d = self.jd(X, p)
        
        #the sampling plan quality criterion
        Phiq = (np.sum(J * ( d**(-q))))**(1.0 / q)
        return Phiq

    def jd(self, X,p=1):
        """
        Computes the distances between all pairs of points in a sampling plan
        X using the p-norm, sorts them in ascending order and removes multiple occurences.

        Inputs:
            X-sampling plan being evaluated
            p-distance norm (p=1 rectangular-default, p=2 Euclidean)
        Output:
            J-multiplicity array (that is, the number of pairs separated by each distance value)
            distinct_d-list of distinct distance values

        """
        #number of points in the sampling plan
        n = np.size(X[:,1])

        #computes the distances between all pairs of points
        d = np.zeros((n*(n-1)//2))



    #    for i in xrange(n-1):
    #        for j in xrange(i+1,n):
    #            if i == 0:
    #                d[i+j-1] = np.linalg.norm((rld[0,:]-rld[j,:]),2)
    #            else:
    #                d[((i-1)*n - (i-1)*i/2 + j - i  )] = np.linalg.norm((X[i,:] - X[j,:]),2)

        #an alternative way of the above loop
        list = [(i,j) for i in range(n-1) for j in range(i+1,n)]
        for k,l in enumerate(list):
            d[k] = np.linalg.norm((X[l[0],:]-X[l[1],:]),p)


        #remove multiple occurences
        distinct_d, J = np.unique(d, return_counts=True)

        return J, distinct_d

    def mm(self,X1,X2,p=1):
        """
        Given two sampling plans chooses the one with the better space-filling properties
        (as per the Morris-Mitchell criterion)

        Inputs:
            X1,X2-the two sampling plans
            p- the distance metric to be used (p=1 rectangular-default, p=2 Euclidean)
        Outputs:
            Mmplan-if Mmplan=0, identical plans or equally space-
            filling, if Mmplan=1, X1 is more space filling, if Mmplan=2,
            X2 is more space filling
        """

        #thats how two arrays are compared in their sorted form
        v = np.sort(X1) == np.sort(X2)
        if 	v.all() == True:#if True, then the designs are the same
    #    if np.array_equal(X1,X2) == True:
            return 0
        else:
            #calculate the distance and multiplicity arrays
            [J1 , d1] = self.jd(X1,p);m1=len(d1)
            [J2 , d2] = self.jd(X2,p);m2=len(d2)

            #blend the distance and multiplicity arrays together for
            #comparison according to definition 1.2B. Note the different
            #signs - we are maximising the d's and minimising the J's.
            V1 = np.zeros((2*m1))
            V1[0:len(V1):2] = d1
            V1[1:len(V1):2] = -J1

            V2 = np.zeros((2*m2))
            V2[0:len(V2):2] = d2
            V2[1:len(V2):2] = -J2

            #the longer vector can be trimmed down to the length of the shorter one
            m = min(m1,m2)
            V1 = V1[0:m]
            V2 = V2[0:m]

            #generate vector c such that c(i)=1 if V1(i)>V2(i), c(i)=2 if V1(i)<V2(i)
            #c(i)=0 otherwise
            c = np.zeros(m)
            for i in range(m):
                if np.greater(V1[i],V2[i]) == True:
                    c[i] = 1
                elif np.less(V1[i],V2[i]) == True:
                    c[i] = 2
                elif np.equal(V1[i],V2[i]) == True:
                    c[i] = 0

            #If the plans are not identical but have the same space-filling
            #properties
            if sum(c) == 0:
                return 0
            else:
                #the more space-filling design (mmplan)
                #is the first non-zero element of c
                i = 0
                while c[i] == 0:
                    i = i+1
                return c[i]


if __name__=='__main__':
    # print fullfactorial2d(2)
    # print fullfactorial3d(2)
    # print fullfactorial4d(2)
    # print fullfactorial5d(2)
    # print optimalLHC()

    sp = samplingplan(k=2)
    print(sp.fullfactorial())
    print(sp.rlh(15))
    print(sp.optimallhc(16))
