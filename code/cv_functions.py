""" Functions for C-fold CV procedure. """

import numpy as np
from sklearn import metrics
import MTCOV as mtcov


def shuffle_indicesG(N, L, rseed=10):
    """
        Extract a maskG using KFold.
    """

    rng = np.random.RandomState(rseed)
    idxG = []
    for a in range(L):
        idx_tmp = []
        for i in range(N):
            for j in range(N):
                idx_tmp.append((i, j))
        idxG.append(idx_tmp)
    for l in range(L):
        rng.shuffle(idxG[l])

    return idxG


def shuffle_indicesX(N, rseed=10):
    """
        Extract a maskX using KFold.
    """

    rng = np.random.RandomState(rseed)
    idxX = np.arange(N)
    rng.shuffle(idxX)

    return idxX


def extract_masks(N, L, idxG=None, idxX=None, cv_type='kfold', NFold=5, fold=0, rseed=10, out_mask=False):
    """
        Return the masks for selecting the held out set in the adjacency tensor and design matrix.

        Parameters
        ----------
        N : int
            Number of nodes.
        L : int
            Number of layers.
        idxG : L-dim list of lists
              Each list has the indexes of the entries of the adjacency matrix of layer L, when cv is set to kfold.
        idxX : list
              List with the indexes of the entries of design matrix, when cv is set to kfold.
        cv_type : string
             Type of cross-validation: kfold or random.
        NFold : int
                Number of C-fold.
        fold : int
               Current fold.
        rseed : int
              Random seed.
        out_mask : bool
                   If set to True, the masks are saved into files.

        Returns
        -------
        maskG : ndarray
                Mask for selecting the held out set in the adjacency tensor.
        maskX : ndarray
                Mask for selecting the held out set in the design matrix.
    """

    maskG = np.zeros((L, N, N), dtype=bool)
    maskX = np.zeros(N, dtype=bool)

    if cv_type == 'kfold':  # sequential order of folds
        # adjacency tensor
        assert L == len(idxG)
        for l in range(L):
            n_samples = len(idxG[l])
            test = idxG[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
            for idx in test:
                maskG[l][idx] = 1

        # design matrix
        testcov = idxX[fold * (N // NFold):(fold + 1) * (N // NFold)]
        maskX[testcov] = 1

    else:  # random split for choosing the test set
        rng = np.random.RandomState(rseed)  # Mersenne-Twister random number generator
        maskG = rng.binomial(1, 1. / float(NFold), size=(L, N, N))
        maskX = rng.binomial(1, 1. / float(NFold), size=N)

    if out_mask:  # output the masks into files
        outmask = '../data/input/mask_f' + str(fold)
        np.savez_compressed(outmask + '.npz', maskG=np.where(maskG > 0.), maskX=np.where(maskX > 0.))
        # To load: mask = np.load('mask_f0.npz'), e.g. print(np.array_equal(maskG, mask['maskG']))
        print('Masks saved in:', outmask)

    return maskG, maskX


def train_running_model(B_cv, X_cv, flag_conv, N, L, C, Z, gamma=0., undirected=False, cv=False, rseed=10, inf=1e10,
                        err_max=0.0000001, err=0.1, N_real=1, tolerance=0.1, decision=10, maxit=500,
                        folder='../data/output/5-fold_cv/', end_file='.csv', assortative=False):
    """
        Run MTCOV model on the train set and return its estimated parameters: U, V, W and beta. U and V are the
        membership matrices, W is the affinity tensor and beta measures the relation between communities and attribute.

        Parameters
        ----------
        B_cv : ndarray
               Graph adjacency tensor.
        X_cv : ndarray
               Object representing the one-hot encoding version of the design matrix.
        flag_conv : str
                    If 'log' the convergence is based on the loglikelihood values; if 'deltas' the convergence is
                    based on the differences in the parameters values. The latter is suggested when the dataset
                    is big (N > 1000 ca.).
        N : int
            Number of nodes.
        L : int
            Number of layers.
        C : int
            Number of communities.
        Z : int
            Number of modalities of the attribute.
        gamma : float
                Scaling parameter gamma.
        undirected : bool
                     If set to True, the network is undirected.
        cv : bool
             If set to True, it performs cross-validation procedure.
        rseed : int
                Random seed for the initialization.
        inf : int
              Initial value for log-likelihood and parameters.
        err_max : float
                  Minimum value for the parameters.
        err : float
              Error for the initialization of W.
        N_real : int
                 Number of iterations with different random initialization.
        tolerance : float
                    Tolerance parameter for convergence.
        decision : int
                   Convergence parameter.
        maxit : int
                Maximum number of EM steps before aborting.
        folder : str
                 Path for storing the output.
        end_file : str
                   Output file suffix.
        assortative : bool
                      If True, the network is assortative.

        Returns
        -------
        U : ndarray
            Membership matrix (out-degree).
        V : ndarray
            Membership matrix (in-degree).
        W : ndarray
            Affinity tensor.
        beta : ndarray
               Beta parameter matrix.
        maxL : float
               Maximum log-likelihood value.
    """

    MTCOV = mtcov.MTCOV(N=N, L=L, C=C, Z=Z, gamma=gamma, undirected=undirected, cv=cv, rseed=rseed, inf=inf,
                        err_max=err_max, err=err, N_real=N_real, tolerance=tolerance, decision=decision,
                        maxit=maxit, folder=folder, end_file=end_file, assortative=assortative)

    return MTCOV.fit(data=B_cv, data_X=X_cv, flag_conv=flag_conv, nodes=None)


def extract_true_label(X, mask=None):
    """
        Extract true label from X.
    """
    if mask is not None:
        return X.iloc[mask > 0].idxmax(axis=1).values
    else:
        return X.idxmax(axis=1).values


def predict_label(X, u, v, beta, mask=None):
    """
        Compute predicted labels.
    """
    if mask is None:
        probs = np.dot((u + v), beta) / 2
        assert (np.round(np.sum(np.sum(probs, axis=1)), 0) == u[mask > 0].shape[0])
        return [X.columns[el] for el in np.argmax(probs, axis=1)]
    else:
        probs = np.dot((u[mask > 0] + v[mask > 0]), beta) / 2
        assert (np.round(np.sum(np.sum(probs, axis=1)), 0) == u[mask > 0].shape[0])
        return [X.iloc[mask > 0].columns[el] for el in np.argmax(probs, axis=1)]


def covariates_accuracy(X, u, v, beta, mask=None):
    """
        Return the accuracy of the attribute prediction, computed as the fraction of corrected classified examples.

        Parameters
        ----------
        X : DataFrame
            Pandas DataFrame object representing the one-hot encoding version of the design matrix.
        u : ndarray
            Membership matrix (out-degree).
        v : ndarray
            Membership matrix (in-degree).
        beta : ndarray
               Beta parameter matrix.
        mask : ndarray
               Mask for selecting a subset of the design matrix.

        Returns
        -------
        acc : float
              Fraction of corrected classified examples.
    """

    true_label = extract_true_label(X, mask=mask)
    pred_label = predict_label(X, u, v, beta, mask=mask)

    acc = metrics.accuracy_score(true_label, pred_label)

    return acc


def expected_Aija(u, v, w):
    M = np.einsum('ik,jq->ijkq', u, v)
    M = np.einsum('ijkq,akq->aij', M, w)
    return M


def fAUC(R, Pos, Neg):
    y = 0.
    bad = 0.
    for m, a in R:
        if (a > 0):
            y += 1
        else:
            bad += y

    AUC = 1. - (bad / (Pos * Neg))
    return AUC


def calculate_AUC(B, u, v, w, mask=None):
    """
        Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
        (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
        (true negative).

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        u : ndarray
            Membership matrix (out-degree).
        v : ndarray
            Membership matrix (in-degree).
        w : ndarray
            Affinity tensor.
        mask : ndarray
               Mask for selecting a subset of the adjacency tensor.

        Returns
        -------
        AUC : float
              AUC value.
    """

    M = expected_Aija(u, v, w)

    if mask is None:
        R = list(zip(M.flatten(), B.flatten()))
        Pos = B.sum()
    else:
        R = list(zip(M[mask > 0], B[mask > 0]))
        Pos = B[mask > 0].sum()

    R.sort(key=lambda x: x[0], reverse=False)
    R_length = len(R)
    Neg = R_length - Pos

    return fAUC(R, Pos, Neg)


def loglikelihood_network(B, u, v, w, mask=None):
    if mask is None:
        M = expected_Aija(u, v, w)
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B * logM).sum() - M.sum()
    else:
        M = expected_Aija(u, v, w)[mask > 0]
        logM = np.zeros(M.shape)
        logM[M > 0] = np.log(M[M > 0])
        return (B[mask > 0] * logM).sum() - M.sum()


def loglikelihood_attributes(X, u, v, beta, mask=None):
    if mask is None:
        p = np.dot(u + v, beta) / 2
        nonzeros = p > 0.
        p[nonzeros] = np.log(p[nonzeros] / 2.)
        return (X * p).sum().sum()
    else:
        p = np.dot(u[mask > 0] + v[mask > 0], beta) / 2
        nonzeros = p > 0.
        p[nonzeros] = np.log(p[nonzeros] / 2.)
        return (X.iloc[mask > 0] * p).sum().sum()


def loglikelihood(B, X, u, v, w, beta, gamma, maskG=None, maskX=None):
    """
        Return the log-likelihood of the model.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        X : DataFrame
            Pandas DataFrame object representing the one-hot encoding version of the design matrix.
        u : ndarray
            Membership matrix (out-degree).
        v : ndarray
            Membership matrix (in-degree).
        w : ndarray
            Affinity tensor.
        beta : ndarray
               Beta parameter matrix.
        gamma : float
                Scaling parameter gamma.
        maskG : ndarray
                Mask for selecting a subset in the adjacency tensor.
        maskX : ndarray
                Mask for selecting a subset in the design matrix.

        Returns
        -------
        loglik : float
                 Log-likelihood value.
    """

    # attribute dimension
    attr = loglikelihood_attributes(X, u, v, beta, mask=maskX)
    # structural dimension
    graph = loglikelihood_network(B, u, v, w, mask=maskG)

    loglik = (1 - gamma) * graph + gamma * attr

    return loglik
