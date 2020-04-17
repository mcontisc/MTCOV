""" K-fold CV technique in conjunction with the Grid Search strategy for estimating the hyper-parameters C and gamma.
Single realization: C and Gamma given in input from command line. """

import os
import time
import csv
import random
import pickle
import numpy as np
from argparse import ArgumentParser
import tools as tl
import cv_functions as cvfun


def main_cv():

    inf = 1e10
    err_max = 0.0000001
    p = ArgumentParser()
    p.add_argument('-j', '--adj_name', type=str, default='adj.csv')
    p.add_argument('-c', '--cov_name', type=str, default='X.csv')
    p.add_argument('-o', '--ego', type=str, default='source')
    p.add_argument('-r', '--alter', type=str, default='target')
    p.add_argument('-x', '--egoX', type=str, default='Name')
    p.add_argument('-a', '--attr_name', type=str, default='Metadata')
    p.add_argument('-C', '--C', type=int, default=2)
    p.add_argument('-g', '--gamma', type=float, default=0.5)
    p.add_argument('-u', '--undirected', type=bool, default=False)
    p.add_argument('-d', '--force_dense', type=bool, default=True)
    p.add_argument('-v', '--cv_type', type=str, choices=['kfold', 'random'], default='kfold')
    p.add_argument('-F', '--NFold', type=int, default=5)
    p.add_argument('-z', '--rseed', type=int, default=107261)
    p.add_argument('-e', '--err', type=float, default=0.1)
    p.add_argument('-i', '--N_real', type=int, default=1)
    p.add_argument('-t', '--tolerance', type=float, default=0.1)
    p.add_argument('-y', '--decision', type=int, default=10)
    p.add_argument('-m', '--maxit', type=int, default=500)
    p.add_argument('-T', '--out_mask', type=int, default=False)
    p.add_argument('-W', '--out_inference', type=int, default=False)
    p.add_argument('-E', '--end_file', type=str, default='_results.csv')
    p.add_argument('-I', '--in_folder', type=str, default='../data/input/')
    p.add_argument('-O', '--out_folder', type=str, default='../data/output/5-fold_cv/')
    args = p.parse_args()

    '''
    Cross validation parameters
    '''
    rseed=args.rseed
    cv_type = args.cv_type  
    NFold = args.NFold  
    rseed = args.rseed
    out_mask = args.out_mask
    out_inference = args.out_inference
    end_file = args.end_file

    '''
    Model parameters
    '''
    C = args.C
    gamma = args.gamma
    N_real = args.N_real

    dataset = args.adj_name.split('.')[0]

    '''
    Set up output directory
    '''
    in_folder = args.in_folder
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    '''
    Import data
    '''
    A, B, X, u_list, v_list = tl.import_data(in_folder, adj_name=args.adj_name, cov_name=args.cov_name, ego=args.ego,
                                             alter=args.alter, egoX=args.egoX, attr_name=args.attr_name,
                                             undirected=args.undirected, force_dense=args.force_dense)

    print('\n### CV procedure ###')
    comparison = [0 for _ in range(10)]
    comparison[0], comparison[1] = C, gamma

    out_file = out_folder+dataset+end_file
    if not os.path.isfile(out_file):  # write header
        with open(out_file, 'w') as outfile:
            wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
            wrtr.writerow(['C', 'gamma', 'fold', 'rseed', 'logL', 'acc_train', 'auc_train', 'logL_test',
                           'acc_test', 'auc_test'])

    time_start = time.time()

    L = B.shape[0]
    N = B.shape[1]
    assert N == X.shape[0]

    if cv_type == 'kfold':
        idxG = cvfun.shuffle_indicesG(N,L,rseed=rseed)
        idxX = cvfun.shuffle_indicesX(N,rseed=rseed)
    else:
        idxG = None
        idxX = None

    with open(out_file, 'a') as outfile:
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
        print('Results will be saved in:', out_file)
        print('\nC =', C, 'gamma =', gamma, '\n')

        for fold in range(NFold):
            print('FOLD ', fold)
            
            ind = rseed+fold  # set the random seed
            comparison[2], comparison[3] = fold, ind

            maskG, maskX = cvfun.extract_masks(N,L,idxG=idxG, idxX=idxX, cv_type=cv_type, NFold=NFold, fold=fold, rseed=ind, out_mask=out_mask)

            '''
            Set up training dataset    
            '''            
            B_train = B.copy()
            B_train[maskG > 0] = 0 

            X_train = X.copy()
            X_train.iloc[maskX > 0] = 0

            '''
            Run MTCOV on the training 
            '''
            tic = time.time()

            U, V, W, BETA, comparison[4] = cvfun.train_running_model(A, B_train, X_train, u_list, v_list,
                                                                     N=A[0].number_of_nodes(),
                                                                     L=len(B),
                                                                     C=C,
                                                                     Z=X.shape[1],
                                                                     gamma=gamma,
                                                                     undirected=args.undirected,
                                                                     cv=True,
                                                                     rseed=rseed,
                                                                     inf=inf,
                                                                     err_max=err_max,
                                                                     err=args.err,
                                                                     N_real=args.N_real,
                                                                     tolerance=args.tolerance,
                                                                     decision=args.decision,
                                                                     maxit=args.maxit,
                                                                     folder=out_folder,
                                                                     end_file=args.end_file)
            
            '''
            Output parameters
            '''
            if out_inference == True:
                outinference = '../data/output/5-fold_cv/theta'+str(fold)+'C'+str(C)+'g'+str(gamma)+'_'+dataset
                np.savez_compressed(outinference+'.npz', u=U, v=V, w=W, beta=BETA) # To load: theta = np.load('test.npz'), e.g. print(np.array_equal(U, theta['u']))
                print('Parameters saved in: ',outinference+'.npz' )
            '''
            Output performance results
            '''
            if gamma != 0:
                comparison[5] = cvfun.covariates_accuracy(X, U, V, BETA, mask=np.logical_not(maskX))
                comparison[8] = cvfun.covariates_accuracy(X, U, V, BETA, mask=maskX)
            if gamma != 1:
                comparison[6] = cvfun.calculate_AUC(B, U, V, W, mask=np.logical_not(maskG))
                comparison[9] = cvfun.calculate_AUC(B, U, V, W, mask=maskG)

            comparison[7] = cvfun.loglikelihood(B, X, U, V, W, BETA, gamma, maskG=maskG, maskX=maskX)

            print("Time elapsed:", np.round(time.time() - tic, 2), " seconds.")

            wrtr.writerow(comparison)
            outfile.flush()

    print("\nTime elapsed:", np.round(time.time() - time_start, 2), " seconds.")


if __name__ == '__main__':
    main_cv()
