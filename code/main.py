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
import sktensor as skt
import yaml


def main():

    p = ArgumentParser()
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input network
    p.add_argument('-j', '--adj_name', type=str, default='adj.csv')  # name of the adjacency tensor
    p.add_argument('-c', '--cov_name', type=str, default='X.csv')  # name of the design matrix
    p.add_argument('-o', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-r', '--alter', type=str, default='target')  # name of the target of the edge
    p.add_argument('-x', '--egoX', type=str, default='Name')  # name of the column with node labels
    p.add_argument('-a', '--attr_name', type=str, default='Metadata')  # name of the attribute to consider
    p.add_argument('-C', '--C', type=int, default=2)  # number of communities
    p.add_argument('-g', '--gamma', type=float, default=0.5)  # scaling hyper parameter
    p.add_argument('-u', '--undirected', type=bool, default=False)  # flag to call the undirected network
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log')  # flag for convergence
    p.add_argument('-d', '--force_dense', type=bool, default=False)  # flag to force a dense transformation in input
    p.add_argument('-b', '--batch_size', type=int, default=None)  # size of the batch used to compute the likelihood
    args = p.parse_args()

    in_folder = args.in_folder

    '''
    Import data
    '''
    A, B, X, nodes = tl.import_data(in_folder, adj_name=args.adj_name, cov_name=args.cov_name, ego=args.ego,
                                    alter=args.alter, egoX=args.egoX, attr_name=args.attr_name,
                                    undirected=args.undirected, force_dense=args.force_dense,
                                    noselfloop=True, verbose=True)

    Xs = np.array(X)
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    if args.batch_size and args.batch_size > len(nodes):
        raise ValueError('The batch size has to be smaller than the number of nodes.')
    if len(nodes) > 1000:
        args.flag_conv = 'deltas'

    # setting to run the algorithm
    with open('setting_MTCOV.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'])
    conf['end_file'] += '_GT'

    '''
    Run MTCOV 
    '''
    print('\n### Run MTCOV ###')
    print(f'Setting: \nC = {args.C}\ngamma = {args.gamma}\n')

    time_start = time.time()
    MTCOV = mtcov.MTCOV(N=A[0].number_of_nodes(), L=len(A), C=args.C, Z=X.shape[1], gamma=args.gamma,
                        undirected=args.undirected, **conf)
    _ = MTCOV.fit(data=B, data_X=Xs, flag_conv=args.flag_conv, nodes=nodes, batch_size=args.batch_size)
    print("\nTime elapsed:", np.round(time.time() - time_start, 2), " seconds.")


if __name__ == '__main__':
    main()

