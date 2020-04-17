import time
import numpy as np
import tools as tl
from termcolor import colored


class MultiTensorCov(object):
    def __init__(self, N=100, L=1, C=2, Z=10, gamma=0.5, undirected=False, cv=False, rseed=107261, inf=1e10,
                 err_max=0.0000001, err=0.1, N_real=1, tolerance=0.1, decision=10, maxit=500, folder='../data/output/',
                 end_file='.dat'):

        self.N = N
        self.L = L
        self.C = C
        self.Z = Z
        self.gamma = gamma
        self.undirected = undirected
        self.cv = cv
        self.rseed = rseed
        self.inf = inf
        self.err_max = err_max
        self.err = err
        self.N_real = N_real
        self.tolerance = tolerance
        self.decision = decision
        self.maxit = maxit
        self.folder = folder
        self.end_file = end_file

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.C), dtype=float)  # membership
        self.v = np.zeros((self.N, self.C), dtype=float)  # membership
        self.w = np.zeros((self.C, self.C, self.L), dtype=float)  # affinity
        self.beta = np.zeros((self.C, self.Z), dtype=float)  # beta matrix

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.C), dtype=float)
        self.v_old = np.zeros((self.N, self.C), dtype=float)
        self.w_old = np.zeros((self.C, self.C, self.L), dtype=float)
        self.beta_old = np.zeros((self.C, self.Z), dtype=float)

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.C), dtype=float)
        self.v_f = np.zeros((self.N, self.C), dtype=float)
        self.w_f = np.zeros((self.C, self.C, self.L), dtype=float)
        self.beta_f = np.zeros((self.C, self.Z), dtype=float)

    def cycle_over_realizations(self, A, B, X, u_list, v_list):
        """
            Performing community detection in multilayer networks considering both the topology of interactions and node
            attributes via EM updates.

            Save the membership matrices U and V, the affinity tensor W and the beta matrix.

            Parameters
            ----------
            A : list
                List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
            B : ndarray
                Graph adjacency tensor.
            X: DataFrame
               Pandas DataFrame object representing the one-hot encoding version of the design matrix.
            u_list : list
                     List of indexes of nodes with non zero out-degree over all layers.
            v_list : list
                     List of indexes of nodes with non zero in-degree over all layers.

        """

        maxL = -1e9  # initialization of the maximum log-likelihood

        nodes = list(A[0].nodes)

        for r in range(self.N_real):

            self._initialize(u_list, v_list)

            self._update_old_variables(u_list, v_list)

            # convergence local variables
            coincide = 0
            convergence = False
            it = 0
            loglik = self.inf
            delta_u = delta_v = delta_beta = delta_w = self.inf

            print('Updating realization {0} ...'.format(r),  end=' ')
            time_start = time.time()
            # --- single step iteration update ---
            while not convergence and it < self.maxit:
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_beta, delta_w = self._update_em(B, X)
                it, loglik, coincide, convergence = self._check_for_convergence(B, X, it, loglik, coincide, convergence)
            print('done!')
            print('r = {0} - logLikelihood = {1} - iterations = {2} - time = {3} seconds'.
                  format(r, loglik, it, np.round(time.time() - time_start, 2)))

            if maxL < loglik:
                self._update_optimal_parameters()
                maxL = loglik
                final_it = it
                conv = convergence
            self.rseed += 1
        # end cycle over realizations

        if final_it == self.maxit and not conv:
            # convergence not reaches
            print(colored('Solution failed to converge in {0} EM steps'.format(self.maxit), 'blue'))
        if self.cv:
            return self.u_f, self.v_f, self.w_f, self.beta_f, maxL
        else:
            self.output_results(maxL, nodes, final_it)

    def _initialize(self, u_list, v_list):
        """
            Random initialization of the parameters U, V, W, beta.

            Parameters
            ----------
            u_list : list
                     List of indexes of nodes with non zero out-degree over all layers.
            v_list : list
                     List of indexes of nodes with non zero in-degree over all layers.
        """

        rng = np.random.RandomState(self.rseed)  # Mersenne-Twister random number generator
        self._randomize_u_v(rng, u_list, v_list)
        if self.gamma != 0:
            self._randomize_beta(rng)
        if self.gamma != 1:
            self._randomize_w(rng)

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor W.

            Parameters
            ----------
            rng : numpy random number generator
        """

        for i in range(self.L):
            for k in range(self.C):
                for q in range(k, self.C):
                    if q == k:
                        self.w[k, q, i] = rng.random_sample(1)
                    else:
                        # the self.err is to impose two different magnitudes to in and off-diagonal elements,
                        # optional, to bias to start in a more 'assortative' or 'disassortative' topology
                        self.w[k, q, i] = self.w[q, k, i] = self.err * rng.random_sample(1)

    def _randomize_beta(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the beta matrix.

            Parameters
            ----------
            rng : numpy random number generator
        """

        for k in range(self.C):
            for z in range(self.Z):
                self.beta[k, z] = rng.random_sample(1)
        # normalization: each row of the matrix beta has to sum to 1
        self.beta = (self.beta.T / np.sum(self.beta, axis=1)).T

    def _randomize_u_v(self, rng, u_list, v_list):
        """
            Assign a random number in (0, 1.) to each entry (with degree greater than zero) of the membership
            matrices U and V.

            Parameters
            ----------
            rng : numpy random number generator
            u_list : list
                     List of indexes of nodes with non zero out-degree over all layers.
            v_list : list
                     List of indexes of nodes with non zero in-degree over all layers.
        """

        for k in range(self.C):
            for i in u_list:
                self.u[i, k] = rng.random_sample(1)
                if self.undirected:
                    self.v[i, k] = self.u[i, k]
            if not self.undirected:
                for j in v_list:
                    self.v[j, k] = rng.random_sample(1)
        # normalization: each row of the membership matrices has to sum to 1
        self.u = (self.u.T / np.sum(self.u, axis=1)).T
        self.v = (self.v.T / np.sum(self.v, axis=1)).T

    def _update_old_variables(self, u_list, v_list):

        for k in range(self.C):
            for i in u_list:
                self.u_old[i, k] = self.u[i, k]
            for j in v_list:
                self.v_old[j, k] = self.v[j, k]
            for l in range(self.L):
                for q in range(self.C):
                        self.w_old[k, q, l] = self.w[k, q, l]
            for z in range(self.Z):
                self.beta_old[k, z] = self.beta[k, z]

    def _update_em(self, B, X):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            B : ndarray
                Graph adjacency tensor.
            X: DataFrame
               Pandas DataFrame object representing the one-hot encoding version of the design matrix.

            Returns
            -------
            d_u : float
                  Maximum distance between the old and the new membership matrix U.
            d_v : float
                  Maximum distance between the old and the new membership matrix V.
            d_beta : float
                     Maximum distance between the old and the new beta matrix.
            d_w : float
                  Maximum distance between the old and the new affinity tensor U.
        """

        if self.gamma == 0:
            d_w = self._update_W(B)
            d_beta = 0.
        elif self.gamma == 1:
            d_beta = self.update_beta(X)
            d_w = 0.
        else:
            d_w = self._update_W(B)
            d_beta = self.update_beta(X)

        d_u = self._update_U(B, X)

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
        else:
            d_v = self._update_V(B, X)

        return d_u, d_v, d_beta, d_w

    def _update_W(self, B):

        uk = np.einsum('ik->k', self.u)
        vk = np.einsum('ik->k', self.v)
        Z_kq = np.einsum('k,q->kq', uk, vk)
        Z_ija = np.einsum('jq,kqa->jka', self.v, self.w_old)

        Z_ija = np.einsum('ik,jka->ija', self.u, Z_ija)

        E = np.einsum('aij->ija', B)
        non_zeros = Z_ija > 0.
        Z_ija[non_zeros] = E[non_zeros] / Z_ija[non_zeros]

        rho_ijkqa = np.einsum('ija,ik->jka', Z_ija, self.u)

        rho_ijkqa = np.einsum('jka,jq->kqa', rho_ijkqa, self.v)
        rho_ijkqa = np.einsum('kqa,kqa->kqa', rho_ijkqa, self.w_old)
        non_zeros = Z_kq > 0.
        # update only non_zeros elements
        for a in range(self.L):
            self.w[non_zeros, a] = rho_ijkqa[non_zeros, a] / Z_kq[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = self.w

        return dist_w

    def update_beta(self, X):

        if self.undirected:
            U = self.u
        else:
            U = self.u + self.v
        denominator = np.einsum('ik,kz->iz', U, self.beta_old)
        nonzeros = denominator > 0.
        denominator[nonzeros] = 1./denominator[nonzeros]
        h_kz = np.einsum('iz, ik, iz->kz', X, U, denominator)
        numerator = self.beta_old * h_kz

        # np.einsum('kz,kz->k', self.beta_old, h_kz) is exactly np.sum(numerator, axis=1)
        non_zeros = np.sum(numerator, axis=1) > 0.
        numerator[non_zeros] /= np.sum(numerator[non_zeros], axis=1, keepdims=True)
        # update only non_zeros elements
        self.beta[non_zeros] = numerator[non_zeros]

        low_values_indices = self.beta < self.err_max  # values are too low
        self.beta[low_values_indices] = 0.  # and set to 0.

        dist_beta = np.amax(abs(self.beta - self.beta_old))
        self.beta_old = self.beta

        return dist_beta

    def _update_U(self, B, X):

        rho_ijka = np.einsum('jq, kqa->jka', self.v, self.w)
        Z_ija = np.einsum('ik,jka->ija', self.u_old, rho_ijka)
        non_zeros = Z_ija > 0.
        E = np.einsum('aij->ija', B)
        Z_ija[non_zeros] = E[non_zeros] / Z_ija[non_zeros]
        graph = np.einsum('ija, jka->ik', Z_ija, rho_ijka)
        graph = self.u_old * graph

        if self.undirected:
            U = self.u_old
        else:
            U = self.u_old + self.v
        denominator = np.einsum('ik,kz->iz', U, self.beta)
        nonzeros = denominator > 0.
        denominator[nonzeros] = 1./denominator[nonzeros]
        h_ik = np.einsum('iz,kz,iz->ik', X, self.beta, denominator)
        attr = U * h_ik

        numerator = self.gamma * attr + (1. - self.gamma) * graph

        # self.gamma + (1 - self.gamma) * np.einsum('ija->i', E) is exactly np.sum(numerator, axis=1)
        non_zeros = np.sum(numerator, axis=1) > 0.
        numerator[non_zeros] /= np.sum(numerator[non_zeros], axis=1, keepdims=True)
        # update only non_zeros elements
        self.u[non_zeros] = numerator[non_zeros]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = self.u

        return dist_u

    def _update_V(self, B, X):

        rho_ijka = np.einsum('ik, kqa->iqa', self.u, self.w)
        Z_jia = np.einsum('jq, iqa->jia', self.v_old, rho_ijka)
        non_zeros = Z_jia > 0.
        E = np.einsum('aij->jia', B)
        Z_jia[non_zeros] = E[non_zeros] / Z_jia[non_zeros]
        graph = np.einsum('jia, iqa->jq', Z_jia, rho_ijka)
        graph = self.v_old * graph

        U = self.u + self.v_old
        denominator = np.einsum('ik,kz->iz', U, self.beta)
        nonzeros = denominator > 0.
        denominator[nonzeros] = 1./denominator[nonzeros]
        h_ik = np.einsum('iz,kz,iz->ik', X, self.beta, denominator)
        attr = U * h_ik

        numerator = self.gamma * attr + (1. - self.gamma) * graph

        # self.gamma + (1 - self.gamma) * np.einsum('jia->j', E)) is exactly np.sum(numerator, axis=1)
        non_zeros = np.sum(numerator, axis=1) > 0.
        numerator[non_zeros] /= np.sum(numerator[non_zeros], axis=1, keepdims=1)
        # update only non_zeros elements
        self.v[non_zeros] = numerator[non_zeros]

        low_values_indices = self.v < self.err_max  # Where values are low
        self.v[low_values_indices] = 0.  # All low values set to 0

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = self.v

        return dist_v

    def _check_for_convergence(self, B, X, it, loglik, coincide, convergence):
        """
            Check for convergence.

            Parameters
            ----------
            B : ndarray
                Graph adjacency tensor.
            X: DataFrame
               Pandas DataFrame object representing the one-hot encoding version of the design matrix.
            it : int
                 Number of iteration.
            loglik : float
                     Loglikelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Loglikelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            loglik = self._Likelihood(B, X)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def _Likelihood(self, B, X):

        U = self.u + self.v

        p_z = np.einsum('ik,kz->iz', U, self.beta)
        nonzeros = p_z > 0.
        p_z[nonzeros] = np.log(p_z[nonzeros]/2.)
        attr = (X * p_z).sum().sum()

        mu_ija = np.einsum('jq, kqa->jka', self.v, self.w)
        mu_ija = np.einsum('ik, jka -> ija', self.u, mu_ija)
        nonzeros = mu_ija > 0.
        M = np.copy(mu_ija)
        M[nonzeros] = np.log(mu_ija[nonzeros])
        E = np.einsum('aij->ija', B)
        Alog = (E * M).sum()
        graph = Alog - mu_ija.sum()

        return np.round(self.gamma * attr + (1. - self.gamma) * graph, 3)

    def _update_optimal_parameters(self):

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.beta_f = np.copy(self.beta)
        self.w_f = np.copy(self.w)

    def output_results(self,maxL, nodes, final_it):
        outinference = self.folder+'theta'+self.end_file
        print('Output in : ',outinference+'.npz')
        np.savez_compressed(outinference+'.npz', u=self.u_f, v=self.v_f, w=self.w_f, beta=self.beta_f, max_it=final_it,nodes=nodes,maxL=maxL,N_real=self.N_real) # To load: theta = np.load('test.npz'), e.g. print(np.array_equal(U, theta['u']))
            
    def output_results2(self, maxL, nodes, X):
        """
            Output results after convergence.

            Parameters
            ----------
            maxL : float
                   Maximum log-likelihood.
            nodes : list
                    List of nodes.
            X: DataFrame
               Pandas DataFrame object representing the one-hot encoding version of the design matrix.
        """
        
        # sort node list if possible
        sorting = tl.can_cast(nodes[0])
        if sorting:
            node_list = np.sort([int(i) for i in nodes])

        infile1 = self.folder + "u" + self.end_file
        in1 = open(infile1, 'w')
        if not self.undirected:
            infile1b = self.folder + "v" + self.end_file
            in1b = open(infile1b, 'w')
        if self.gamma != 0:
            infile2 = self.folder + "beta" + self.end_file
            in2 = open(infile2, 'w')
        if self.gamma != 1:
            infile3 = self.folder + "w" + self.end_file
            in3 = open(infile3, 'w')

        print("# Max Likelihood = {0} - N_real = {1} - gamma = {2}\n".format(maxL, self.N_real, self.gamma), file=in1)
        if not self.undirected:
            print("# Max Likelihood = {0} - N_real = {1} - gamma = {2}\n".format(maxL, self.N_real, self.gamma),
                  file=in1b)
        if self.gamma != 0:
            print("# Max Likelihood = {0} - N_real = {1} - gamma = {2}\n".format(maxL, self.N_real, self.gamma),
                  file=in2)
        if self.gamma != 1:
            print("# Max Likelihood = {0} - N_real = {1} - gamma = {2}\n".format(maxL, self.N_real, self.gamma),
                  file=in3)

        # Output membership matrices
        if sorting:
            for u in node_list:
                try:
                    i = nodes.index(str(u))
                except ValueError:
                    i = nodes.index(u)
                print(u, file=in1, end=' ')
                if not self.undirected:
                    print(u, file=in1b, end=' ')
                for k in range(self.C):
                    print(self.u_f[i][k], file=in1, end=' ')
                    if not self.undirected:
                        print(self.v_f[i][k], file=in1b, end=' ')
                print(file=in1)
                if not self.undirected:
                    print(file=in1b)
        else:
            for i in range(self.N):
                print(nodes[i], file=in1, end=' ')
                if not self.undirected:
                    print(nodes[i], file=in1b, end=' ')
                for k in range(self.C):
                    print(self.u_f[i][k], file=in1, end=' ')
                    if not self.undirected:
                        print(self.v_f[i][k], file=in1b, end=' ')
                print(file=in1)
                if not self.undirected:
                    print(file=in1b)
        in1.close()
        if not self.undirected:
            in1b.close()

        # Output affinity tensor
        if self.gamma != 1:
            for l in range(self.L):
                print("a=", l, file=in3)
                for k in range(self.C):
                    for q in range(self.C):
                        print(self.w_f[k][q][l], file=in3, end=' ')
                    print(file=in3)
                print('\n', file=in3)
            in3.close()

        # Output beta matrix
        if self.gamma != 0:
            for col in X.columns:
                print(col, file=in2, end=' ')
            print(file=in2)
            for k in range(self.C):
                for z in range(self.Z):
                    print(self.beta_f[k][z], file=in2, end=' ')
                print(file=in2)
            in2.close()

        print("\nData saved in: ")
        print(infile1)
        if not self.undirected:
            print(infile1b)
        if self.gamma != 0:
            print(infile2)
        if self.gamma != 1:
            print(infile3)

