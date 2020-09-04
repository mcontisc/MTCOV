from __future__ import print_function
import time
import sys
import sktensor as skt
import numpy as np
import scipy.sparse
from termcolor import colored


class MTCOV:
    def __init__(self, N=100, L=1, C=2, Z=10, gamma=0.5, undirected=False, cv=False, rseed=107261, inf=1e10,
                 err_max=0.0000001, err=0.1, N_real=1, tolerance=0.001, decision=10, maxit=500, folder='../data/output/',
                 end_file='.dat', assortative=False):

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
        self.assortative = assortative

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.C), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.C), dtype=float)  # in-going membership
        self.beta = np.zeros((self.C, self.Z), dtype=float)  # beta matrix

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.C), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.C), dtype=float)  # in-going membership
        self.beta_old = np.zeros((self.C, self.Z), dtype=float)  # beta matrix

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.C), dtype=float)  # out-going membership
        self.v_f = np.zeros((self.N, self.C), dtype=float)  # in-going membership
        self.beta_f = np.zeros((self.C, self.Z), dtype=float)  # beta matrix

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.C), dtype=float)
            self.w_old = np.zeros((self.L, self.C), dtype=float)
            self.w_f = np.zeros((self.L, self.C), dtype=float)
        else:
            self.w = np.zeros((self.L, self.C, self.C), dtype=float)
            self.w_old = np.zeros((self.L, self.C, self.C), dtype=float)
            self.w_f = np.zeros((self.L, self.C, self.C), dtype=float)

    def fit(self, data, data_X, flag_conv, nodes):
        """
            Performing community detection in multilayer networks considering both the topology of interactions and node
            attributes via EM updates.
            Save the membership matrices U and V, the affinity tensor W and the beta matrix.

            Parameters
            ----------
            data : ndarray
                   Graph adjacency tensor.
            data_X : ndarray
                    Object representing the one-hot encoding version of the design matrix.
            flag_conv : str
                        If 'log' the convergence is based on the loglikelihood values; if 'deltas' the convergence is
                        based on the differences in the parameters values. The latter is suggested when the dataset
                        is big (N > 1000 ca.).
            nodes : list
                    List of nodes IDs.
        """

        maxL = -1e9  # initialization of the maximum log-likelihood

        # pre-processing of the data to handle the sparsity
        data = preprocess(data)
        data_X = preprocess_X(data_X)
        # save the indexes of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs
        subs_X_nz = data_X.nonzero()

        rng = np.random.RandomState(self.rseed)

        for r in range(self.N_real):

            self._initialize(rng=np.random.RandomState(self.rseed))

            self._update_old_variables()
            self._update_cache(data, subs_nz, data_X, subs_X_nz)

            # convergence local variables
            coincide, it = 0, 0
            convergence = False
            if flag_conv == 'log':
                loglik = self.inf

            print('Updating realization {0} ...'.format(r),  end=' ')
            time_start = time.time()
            # --- single step iteration update ---
            while not convergence and it < self.maxit:
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w, delta_beta = self._update_em_compact(data, data_X, subs_nz, subs_X_nz)
                if flag_conv == 'log':
                    it, loglik, coincide, convergence = self._check_for_convergence(data, data_X, it, loglik, coincide,
                                                                                convergence)
                elif flag_conv == 'deltas':
                    it, coincide, convergence = self._check_for_convergence_delta(it, coincide, delta_u, delta_v,
                                                                                  delta_w, delta_beta, convergence)
                else:
                    print(colored('Error! flag_conv can be either "log" or "deltas"', 'red'))
                    break
            print('done!')
            print('r = {0} - iterations = {1} - time = {2} seconds'. format(r, it, np.round(time.time() - time_start, 2)))

            if flag_conv == 'log':
                if maxL < loglik:
                    self._update_optimal_parameters()
                    maxL = loglik
                    final_it = it
                    conv = convergence
            elif flag_conv == 'deltas':
                self._update_optimal_parameters()
                final_it = it
                conv = convergence
            self.rseed += rng.randint(10000)
        # end cycle over realizations

        if final_it == self.maxit and not conv:
            # convergence not reaches
            print(colored('Solution failed to converge in {0} EM steps'.format(self.maxit), 'blue'))

        if self.cv:
            return self.u_f, self.v_f, self.w_f, self.beta_f, maxL
        else:
            self.output_results(maxL, nodes, final_it)

    def _initialize(self, rng):
        """
            Random initialization of the parameters U, V, W, beta.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        self._randomize_u_v(rng)
        if self.gamma != 0:
            self._randomize_beta(rng)
        if self.gamma != 1:
            self._randomize_w(rng)

    def _randomize_u_v(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices U and V, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        self.u = rng.random_sample(self.u.shape)
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        if not self.undirected:
            self.v = rng.random_sample(self.v.shape)
            row_sums = self.v.sum(axis=1)
            self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        else:
            self.v = self.u

    def _randomize_beta(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the beta matrix, and normalize each row.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """
        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for k in range(self.C):
            for z in range(self.Z):
                self.beta[k, z] = rng.random_sample(1)
        # normalization: each row of the matrix beta has to sum to 1
        self.beta = (self.beta.T / np.sum(self.beta, axis=1)).T

    def _randomize_w(self, rng):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor W.

            Parameters
            ----------
            rng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.C):
                if self.assortative:
                    self.w[i, k] = rng.random_sample(1)
                else:
                    for q in range(k, self.C):
                        if q == k:
                            self.w[i, k, q] = rng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old = self.u.copy()
        self.v_old = self.v.copy()
        self.w_old = self.w.copy()
        self.beta_old = self.beta.copy()

    def _update_cache(self, data, subs_nz, data_X, subs_X_nz):
        """
            Update the cache used in the em_update.

            Parameters
            ----------
            data : ndarray
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            data_X : ndarray
                    Object representing the one-hot encoding version of the design matrix.
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.
        """

        # A
        self.lambda0_nz = self._lambda0_nz(subs_nz, self.u, self.v, self.w)
        self.lambda0_nz[self.lambda0_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.lambda0_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.lambda0_nz

        # X
        self.pi0_nz = self._pi0_nz(subs_X_nz, self.u, self.v, self.beta)
        self.pi0_nz[self.pi0_nz == 0] = 1
        if not scipy.sparse.issparse(data_X):
            self.data_pi_nz = data_X[subs_X_nz[0]] / self.pi0_nz
        else:
            self.data_pi_nz = data_X.data / self.pi0_nz

    def _lambda0_nz(self, subs_nz, u, v, w):
        """
            Compute the mean lambda0 (M_ij^alpha) for only non-zero entries (denominator of pijkl).

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            nz_recon_I : ndarray
                         Mean lambda0 (M_ij^alpha) for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

        return nz_recon_I

    def _pi0_nz(self, subs_X_nz, u, v, beta):
        """
            Compute the mean pi0 (pi_iz) for only non-zero entries (denominator of hizk).

            Parameters
            ----------
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            beta : ndarray
                   Beta matrix.

            Returns
            -------
            Mean pi0 (pi_iz) for only non-zero entries.
        """

        if self.undirected:
            return np.einsum('Ik,kz->Iz', u[subs_X_nz[0], :], beta)
        else:
            return np.einsum('Ik,kz->Iz', u[subs_X_nz[0], :] + v[subs_X_nz[0], :], beta)

    def _update_em_compact(self, data, data_X, subs_nz, subs_X_nz):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : ndarray
                   Graph adjacency tensor.
            data_X : ndarray
                     Object representing the one-hot encoding version of the design matrix.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.

            Returns
            -------
            d_u : float
                  Maximum distance between the old and the new membership matrix U.
            d_v : float
                  Maximum distance between the old and the new membership matrix V.
            d_beta : float
                     Maximum distance between the old and the new beta matrix.
            d_w : float
                  Maximum distance between the old and the new affinity tensor W.
        """

        if self.gamma < 1.:
            if not self.assortative:
                d_w = self._update_W(subs_nz)
            else:
                d_w = self._update_W_assortative(subs_nz)
        else:
            d_w = 0
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        if self.gamma > 0.:
            d_beta = self._update_beta(subs_X_nz)
        else:
            d_beta = 0.
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        d_u = self._update_U(subs_nz, subs_X_nz)
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        if self.undirected:
            self.v = self.u
            self.v_old = self.v
            d_v = d_u
        else:
            d_v = self._update_V(subs_nz, subs_X_nz)
        self._update_cache(data, subs_nz, data_X, subs_X_nz)

        return d_u, d_v, d_w, d_beta

    def _update_W(self, subs_nz):
        """
            Update affinity tensor.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor W.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV

        for k in range(self.C):
            for q in range(self.C):
                uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))
        non_zeros = Z > 0

        for a in range(self.L):
            self.w[a, non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz):
        """
            Update affinity tensor (assuming assortativity).

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.

            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor W.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = self.data_M_nz[:, np.newaxis] * UV

        for k in range(self.C):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))
        non_zeros = Z > 0
        for a in range(self.L):
            self.w[a, non_zeros] /= Z[non_zeros]

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_beta(self, subs_X_nz):
        """
            Update beta matrix.

            Parameters
            ----------
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.

            Returns
            -------
            dist_beta : float
                        Maximum distance between the old and the new beta matrix.
        """

        if self.undirected:
            XUV = np.einsum('Iz,Ik->kz', self.data_pi_nz, self.u[subs_X_nz[0], :])
        else:
            XUV = np.einsum('Iz,Ik->kz', self.data_pi_nz, self.u[subs_X_nz[0], :] + self.v[subs_X_nz[0], :])
        self.beta *= XUV

        row_sums = self.beta.sum(axis=1)
        self.beta[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.beta < self.err_max  # values are too low
        self.beta[low_values_indices] = 0.  # and set to 0.

        dist_beta = np.amax(abs(self.beta - self.beta_old))
        self.beta_old = np.copy(self.beta)

        return dist_beta

    def _update_U(self, subs_nz, subs_X_nz):
        """
            Update out-going membership matrix.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data_X that are non-zero.
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.

            Returns
            -------
            dist_u : float
                     Maximum distance between the old and the new membership matrix U.
        """

        self.u = self._update_membership(subs_nz, subs_X_nz, self.u, self.v, self.w, self.beta, 1)

        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self, subs_nz, subs_X_nz):
        """
            Update in-coming membership matrix.
            Same as _update_U but with:
            data <-> data_T
            w <-> w_T
            u <-> v

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data_X that are non-zero.
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.

            Returns
            -------
            dist_v : float
                     Maximum distance between the old and the new membership matrix V.
        """

        self.v = self._update_membership(subs_nz, subs_X_nz, self.u, self.v, self.w, self.beta, 2)

        row_sums = self.v.sum(axis=1)
        self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_membership(self, subs_nz, subs_X_nz, u, v, w, beta, m):
        """
            Main procedure to update membership matrices.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            subs_X_nz : tuple
                        Indices of elements of data_X that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            beta : ndarray
                   Beta matrix.
            m : int
                Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
                works with the matrix U; if 2 it works with V.

            Returns
            -------
            out : ndarray
                  Update of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(self.data_M_nz, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz, subs_nz, m, u, v, w)

        if m == 1:
            uttkrp_DK *= u
        elif m == 2:
            uttkrp_DK *= v

        uttkrp_Xh = np.einsum('Iz,kz->Ik', self.data_pi_nz, beta)

        if self.undirected:
            uttkrp_Xh *= u[subs_X_nz[0]]
        else:
            uttkrp_Xh *= u[subs_X_nz[0]] + v[subs_X_nz[0]]

        uttkrp_DK *= (1 - self.gamma)
        out = uttkrp_DK.copy()
        out[subs_X_nz[0]] += self.gamma * uttkrp_Xh

        return out

    def _check_for_convergence(self, data, data_X, it, loglik, coincide, convergence):
        """
            Check for convergence by using the log-likelihood values.

            Parameters
            ----------
            data : ndarray
                   Graph adjacency tensor.
            data_X : ndarray
                    Object representing the one-hot encoding version of the design matrix.
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % 10 == 0:
            old_L = loglik
            loglik = self.__Likelihood(data, data_X)
            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def __Likelihood(self, data, data_X):
        """
            Compute the log-likelihood of the data.

            Parameters
            ----------
            data : ndarray
                   Graph adjacency tensor.
            data_X : ndarray
                    Object representing the one-hot encoding version of the design matrix.

            Returns
            -------
            l : float
                Log-likelihood value.
        """

        self.lambda0_ija = self._lambda0_full(self.u, self.v, self.w)
        lG = -self.lambda0_ija.sum()
        logM = np.log(self.lambda0_nz)
        if isinstance(data, skt.dtensor):
            Alog = data[data.nonzero()] * logM
        elif isinstance(data, skt.sptensor):
            Alog = data.vals * logM
        lG += Alog.sum()

        if self.undirected:
            logP = np.log(self.pi0_nz)
        else:
            logP = np.log(0.5 * self.pi0_nz)
        if not scipy.sparse.issparse(data_X):
            ind_logP_nz = (np.arange(len(logP)), data_X.nonzero()[1])
            Xlog = data_X[data_X.nonzero()] * logP[ind_logP_nz]
        else:
            Xlog = data_X.data * logP
        lX = Xlog.sum()

        l = (1. - self.gamma) * lG + self.gamma * lX

        if np.isnan(l):
            print("Likelihood is NaN!!!!")
            sys.exit(1)
        else:
            return l

    def _lambda0_full(self, u, v, w):
        """
            Compute the mean M_ij^alpha for all entries.

            Parameters
            ----------
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.

            Returns
            -------
            M : ndarray
                Mean M_ij^alpha for all entries.
        """

        if w.ndim == 2:
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)

        return M

    def _check_for_convergence_delta(self, it, coincide, du, dv, dw, db, convergence):
        """
            Check for convergence by using the maximum distances between the old and the new parameters values.

            Parameters
            ----------
            it : int
                 Number of iteration.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            du : float
                 Maximum distance between the old and the new membership matrix U.
            dv : float
                 Maximum distance between the old and the new membership matrix V.
            dw : float
                 Maximum distance between the old and the new affinity tensor W.
            db : float
                 Maximum distance between the old and the new beta matrix.
            convergence : bool
                          Flag for convergence.

            Returns
            -------
            it : int
                 Number of iteration.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if du < self.tolerance and dv < self.tolerance and dw < self.tolerance and db < self.tolerance:
            coincide += 1
        else:
            coincide = 0

        if coincide > self.decision:
            convergence = True
        it += 1

        return it, coincide, convergence

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.beta_f = np.copy(self.beta)

    def output_results(self, maxL, nodes, final_it):
        """
            Output results.

            Parameters
            ----------
            maxL : float
                   Maximum log-likelihood.
            nodes : list
                    List of nodes IDs.
            final_it : int
                       Total number of iterations.
        """

        outinference = self.folder + 'theta' + self.end_file
        print('Output in : ', outinference + '.npz')
        np.savez_compressed(outinference + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, beta=self.beta_f,
                            max_it=final_it, nodes=nodes, maxL=maxL, N_real=self.N_real)
        # to load: theta = np.load('test.npz'), e.g. print(np.array_equal(U, theta['u']))


def sp_uttkrp(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version).

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix U; if 2 it works with V.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        beta : ndarray
               Beta matrix.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating U
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating V
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.

        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix U; if 2 it works with V.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        beta : ndarray
               Beta matrix.

        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating U
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
        elif m == 2:  # we are updating V
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def preprocess(A):
    """
        Pre-process input data tensor.
        If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

        Parameters
        ----------
        A : ndarray
            Input data (tensor).

        Returns
        -------
        A : sptensor/dtensor
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not A.dtype == np.dtype(int).type:
        A = A.astype(int)
    if isinstance(A, np.ndarray) and is_sparse(A):
        A = sptensor_from_dense_array(A)
    else:
        A = skt.dtensor(A)
    return A


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It Implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def preprocess_X(X):
    """
        Pre-process input data matrix.
        If the input is sparse, returns a scipy sparse matrix. Otherwise, returns a numpy array.

        Parameters
        ----------
        X : ndarray
            Input data (matrix).

        Returns
        -------
        X : scipy sparse matrix/ndarray
            Pre-processed data. If the input is sparse, returns a scipy sparse matrix. Otherwise, returns a numpy array.
    """

    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if isinstance(X, np.ndarray) and scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(X)

    return X