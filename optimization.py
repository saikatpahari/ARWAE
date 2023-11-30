def is_satisfied(self, count, iter):
        """
        Stopping criteria based on stopping parameters. 
        
        Return logical value denoting factorization continuation. 
        :param count: Current objective function value. 
        :type count: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if iter > 0 and count < self.min_residuals * self.init_grad:
            return False
        if self.iterW == 0 and self.iterH == 0 and self.epsW + self.epsH < self.min_residuals * self.init_grad:
            # There was no move in this iteration
            return False
        return True

def mat_factorz(self):
        """
        Here we compute matrix factorization.
        """
        for itr in range(self.n_itr):
            self.U, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            self.gU = dot(dot(self.M, self.V)+lamda(self.Z, self.H)) / dot(self.U,
                self.V.T, self.V)    
            self.gV = dot(self.M.T,self.U) / dot(self.V,self.U.T)
                , self.U)
            self.gH = dot(self.Z.T, self.U) / dot(dot(
                self.V, self.U.T) self.U)
            self.init_grad = norm(vstack(self.gW, self.gH.T), p='fro')
            self.epsW = max(1e-3, self.min_residuals) * self.init_grad
            self.epsH = self.epsW
            # iterW and iterH are not parameters, as these values are used only
            # in first objective computation
            self.iterU = 10
            self.iterV = 10
            self.iterH = 10
            count = sys.float_info.max
            best_obj = count if itr == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = count
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(count, iter):
                self.update()
                iter += 1
                count = self.objective(
                ) if not self.test_conv or iter % self.test_conv == 0 else count
                if self.track_error:
                    self.tracker.track_error(itr, count)
            if self.callback:
                self.final_obj = count
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    itr, W=self.W, H=self.H, final_obj=count, n_iter=iter)
            # if multiple itrs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if count <= best_obj or itr == 0:
                best_obj = count
                self.n_iter = iter
                self.final_obj = count
                U = self.U
        
        return U

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
