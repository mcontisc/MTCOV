"""
    Performing community detection in multilayer networks considering both the topology of interactions and node
    attributes. Implementation of MTCOV algorithm.
"""

import os
import time
import numpy as np
from argparse import ArgumentParser
import tools as tl
import MTCOV as mtcov


def main():
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
    p.add_argument('-z', '--rseed', type=int, default=107261)
    p.add_argument('-e', '--err', type=float, default=0.1)
    p.add_argument('-i', '--N_real', type=int, default=1)
    p.add_argument('-t', '--tolerance', type=float, default=0.1)
    p.add_argument('-y', '--decision', type=int, default=10)
    p.add_argument('-m', '--maxit', type=int, default=500)
    p.add_argument('-E', '--end_file', type=str, default='.dat')
    p.add_argument('-I', '--in_folder', type=str, default='../data/input/')
    p.add_argument('-O', '--out_folder', type=str, default='../data/output/')
    args = p.parse_args()

    time_start = time.time()

    in_folder = args.in_folder
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # run MTCOV
    A, B, X, u_list, v_list = tl.import_data(in_folder, adj_name=args.adj_name, cov_name=args.cov_name, ego=args.ego,
                                             alter=args.alter, egoX=args.egoX, attr_name=args.attr_name, 
                                             undirected=args.undirected, force_dense=args.force_dense)

    print('\n### Run MTCOV ###')
    print(f'Setting: \nC = {args.C}\ngamma = {args.gamma}\nundirected = {args.undirected}\n')

    MTCOV = mtcov.MultiTensorCov(N=A[0].number_of_nodes(),  # number of nodes
                                 L=len(B),  # number of layers
                                 C=args.C,  # number of communities
                                 Z=X.shape[1],  # number of modalities of the attribute
                                 gamma=args.gamma,  # scaling parameter gamma
                                 undirected=args.undirected,  # if True, the network is undirected
                                 rseed=args.rseed,  # random seed for the initialization
                                 inf=inf,  # initial value for log-likelihood and parameters
                                 err_max=err_max,  # minimum value for the parameters
                                 err=args.err,  # error for the initialization of W
                                 N_real=args.N_real,  # number of iterations with different random initialization
                                 tolerance=args.tolerance,  # tolerance parameter for convergence
                                 decision=args.decision,  # convergence parameter
                                 maxit=args.maxit,  # maximum number of EM steps before aborting
                                 folder=out_folder,  # path for storing the output
                                 end_file=args.end_file  # output file suffix
                                 )

    _ = MTCOV.cycle_over_realizations(A, B, X, u_list, v_list)

    print("\nTime elapsed:", np.round(time.time() - time_start, 2), " seconds.")


if __name__ == '__main__':
    main()

