"""
    C-fold CV technique in conjunction with the Grid Search strategy for estimating the hyper-parameters C and gamma.
    Single realization: C and gamma given in input from command line.
"""

# TODO: optimize for big matrices (so when the input would be done with force_dense=False)

import os
import time
import csv
import numpy as np
from argparse import ArgumentParser
import tools as tl
import cv_functions as cvfun
import sktensor as skt
import yaml


def main_cv():

    p = ArgumentParser()
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input network
    p.add_argument('-j', '--adj_name', type=str, default='adj_cv.csv')  # name of the adjacency tensor
    p.add_argument('-c', '--cov_name', type=str, default='X_cv.csv')  # name of the design matrix
    p.add_argument('-o', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-r', '--alter', type=str, default='target')  # name of the target of the edge
    p.add_argument('-x', '--egoX', type=str, default='Name')  # name of the column with node labels
    p.add_argument('-a', '--attr_name', type=str, default='Metadata')  # name of the attribute to consider
    p.add_argument('-C', '--C', type=int, default=2)  # number of communities
    p.add_argument('-g', '--gamma', type=float, default=0.5)  # scaling hyper parameter
    p.add_argument('-u', '--undirected', type=bool, default=True)  # flag to call the undirected network
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log')  # flag for convergence
    # p.add_argument('-d', '--force_dense', type=bool, default=False)  # flag to force a dense transformation in input
    p.add_argument('-b', '--batch_size', type=int, default=None)  # size of the batch to use to compute the likelihood
    p.add_argument('-v', '--cv_type', type=str, choices=['kfold', 'random'], default='kfold')  # type of CV to use
    p.add_argument('-NF', '--NFold', type=int, default=5)  # number of fold to perform cross-validation
    p.add_argument('-T', '--out_mask', type=int, default=False)  # flag to output the masks
    p.add_argument('-or', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
    args = p.parse_args()

    # setting to run the algorithm
    with open('setting_MTCOV.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf['out_folder'] = '../data/output/5-fold_cv/'
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'])

    force_dense = True

    '''
    Cross validation parameters
    '''
    cv_type = args.cv_type
    NFold = args.NFold
    prng = np.random.RandomState(seed=conf['rseed'])  # set seed random number generator
    rseed = prng.randint(1000)
    out_mask = args.out_mask
    out_results = args.out_results

    '''
    Model parameters
    '''
    C = args.C
    gamma = args.gamma

    '''
    Set up directories
    '''
    in_folder = args.in_folder
    out_folder = conf['out_folder']

    dataset = args.adj_name.split('.')[0]

    '''
    Import data
    '''
    A, B, X, nodes = tl.import_data(in_folder, adj_name=args.adj_name, cov_name=args.cov_name, ego=args.ego,
                                    alter=args.alter, egoX=args.egoX, attr_name=args.attr_name,
                                    undirected=args.undirected, force_dense=force_dense,
                                    noselfloop=True, verbose=True)

    Xs = np.array(X)
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    if args.batch_size and args.batch_size > len(nodes):
        raise ValueError('The batch size has to be smaller than the number of nodes.')
    if len(nodes) > 1000:
        args.flag_conv = 'deltas'

    print('\n### CV procedure ###')
    comparison = [0 for _ in range(10)]
    comparison[0], comparison[1] = C, gamma

    # save the results
    if out_results:
        out_file = out_folder+dataset+'_results.csv'
        if not os.path.isfile(out_file):  # write header
            with open(out_file, 'w') as outfile:
                wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
                wrtr.writerow(['C', 'gamma', 'fold', 'rseed', 'logL', 'acc_train', 'auc_train', 'logL_test',
                               'acc_test', 'auc_test'])
        outfile = open(out_file, 'a')
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
        print(f'Results will be saved in: {out_file}')

    time_start = time.time()

    L = B.shape[0]
    N = B.shape[1]
    assert N == X.shape[0]

    '''
    Extract masks
    '''
    if cv_type == 'kfold':
        idxG = cvfun.shuffle_indicesG(N, L, rseed=rseed)
        idxX = cvfun.shuffle_indicesX(N, rseed=rseed)
    else:
        idxG = None
        idxX = None

    print('\nC =', C, 'gamma =', gamma, '\n')

    for fold in range(NFold):
        print('FOLD ', fold)

        rseed += prng.randint(500)
        comparison[2], comparison[3] = fold, rseed

        maskG, maskX = cvfun.extract_masks(N, L, idxG=idxG, idxX=idxX, cv_type=cv_type, NFold=NFold, fold=fold,
                                           rseed=rseed, out_mask=out_mask)

        '''
        Set up training dataset    
        '''
        B_train = B.copy()
        B_train[maskG > 0] = 0

        X_train = Xs.copy()
        X_train[maskX > 0] = 0

        conf['end_file'] = 'GT'+str(fold)+'C'+str(C)+'g'+str(gamma)+'_'+dataset

        '''
        Run MTCOV on the training 
        '''
        tic = time.time()
        U, V, W, BETA, comparison[4] = cvfun.train_running_model(B_cv=B_train, X_cv=X_train, flag_conv=args.flag_conv,
                                                                 C=C, Z=X.shape[1], gamma=gamma, undirected=args.undirected,
                                                                 nodes=nodes, batch_size=args.batch_size, **conf)

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

        if out_results:
            wrtr.writerow(comparison)
            outfile.flush()
    if out_results:
        outfile.close()

    print("\nTime elapsed:", np.round(time.time() - time_start, 2), " seconds.")


if __name__ == '__main__':
    main_cv()
