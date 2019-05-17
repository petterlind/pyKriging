def _reduced_likelihood_function(self, theta):

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
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            return reduced_likelihood_function_value, par
        
        # Get generalized least squares solution
        
       
        Ft = linalg.solve_triangular(C, self.F, lower=True)

        
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

        rho = Yt - np.dot(Ft, beta)
        
        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2. / self.nt)).prod()

        # Compute/Organize output
        if self.name == 'MFK':
            n_samples = self.nt
            p = self.p
            q = self.q
            sigma2 = (rho ** 2.).sum(axis=0) /(n_samples - p - q)
            reduced_likelihood_function_value = -(n_samples - p - q)*np.log10(sigma2) \
                    - n_samples*np.log10(detR)
        else:
            # Bouhlel
            sigma2 = (rho ** 2.).sum(axis=0) / (self.nt)
            pdb.set_trace()
            
            
            reduced_likelihood_function_value = - sigma2.sum() * detR
            # reduced_likelihood_function_value = - 0.5 * (self.nt * float(np.log(sigma2)) + np.log(detR))
            
        # Bouhlel
        par['sigma2'] = sigma2 * self.y_std ** 2.
        
        par['beta'] = beta
        par['gamma'] = linalg.solve_triangular(C.T, rho)
        par['C'] = C
        par['Ft'] = Ft
        par['G'] = G