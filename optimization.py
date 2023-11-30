def is_satisfied(self, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping
        parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param c_obj: Current objective function value. 
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if iter > 0 and c_obj < self.min_residuals * self.init_grad:
            return False
        if self.iterW == 0 and self.iterH == 0 and self.epsW + self.epsH < self.min_residuals * self.init_grad:
            # There was no move in this iteration
            return False
        return True

def factorize(self):
        """
        Here we compute matrix factorization.
        """
        for run in range(self.n_run):
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.gW = dot(self.W, dot(self.H, self.H.T)) - dot(
                self.V, self.H.T)
            self.gH = dot(dot(self.W.T, self.W), self.H) - dot(
                self.W.T, self.V)
            self.init_grad = norm(vstack(self.gW, self.gH.T), p='fro')
            self.epsW = max(1e-3, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            # iterW and iterH are not parameters, as these values are used only
            # in first objective computation
            self.iterW = 10
            self.iterH = 10
            c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(c_obj, iter):
                self.update()
                iter += 1
                c_obj = self.objective(
                ) if not self.test_conv or iter % self.test_conv == 0 else c_obj
                if self.track_error:
                    self.tracker.track_error(run, c_obj)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    run, W=self.W, H=self.H, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

def update(self):
        """Update basis and mixture matrix."""
        self.W, self.gW, self.iterW = self._subproblem(
            self.V.T, self.H.T, self.W.T, self.epsW)
        self.W = self.W.T
        self.gW = self.gW.T
        self.epsW = 0.1 * self.epsW if self.iterW == 0 else self.epsW
        self.H, self.gH, self.iterH = self._subproblem(
            self.V, self.W, self.H, self.epsH)
        self.epsH = 0.1 * self.epsH if self.iterH == 0 else self.epsH
