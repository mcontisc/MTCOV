
import unittest
import numpy as np
import pandas as pd

import MTCOV as mtcov
import tools as tl


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    N = 100
    L = 2
    C = 2
    gamma = 0.5 
    in_folder = '../data/input/'
    out_folder = '../data/output/test/'
    end_file = '_test'
    adj_name = 'adj.csv'
    cov_name = 'X.csv'
    ego = 'source'
    alter = 'target'
    egoX = 'Name'
    attr_name = 'Metadata'
    rseed = 107261
    N_real = 1

    '''
    Import data
    '''
    A, B, X, u_list, v_list = tl.import_data(in_folder, adj_name=adj_name, cov_name=cov_name, ego=ego,
                                             alter=alter, egoX=egoX, attr_name=attr_name)

    MTCOV = mtcov.MultiTensorCov(N=A[0].number_of_nodes(),  # number of nodes
                             L=len(B),  # number of layers
                             C=C,  # number of communities
                             Z=X.shape[1],  # number of modalities of the attribute
                             gamma=gamma,  # scaling parameter gamma
                             rseed=rseed,  # random seed for the initialization
                             N_real=N_real,  # number of iterations with different random initialization
                             folder=out_folder,  # path for storing the output
                             end_file=end_file  # output file suffix
                             )

    # test case function to check the mtcov.set_name function
    def test_import_data(self):
        print("Start import data test\n")
        self.assertTrue(self.B.sum()>0)
        print('B has ',self.B.sum(), ' total weight.')

    # test case function to check the Person.get_name function
    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")
        
        _ = self.MTCOV.cycle_over_realizations(self.A, self.B, self.X, self.u_list, self.v_list)

        theta = np.load(self.MTCOV.folder+'theta'+self.MTCOV.end_file+'.npz')
        thetaGT = np.load(self.MTCOV.folder+'theta_test_GT.npz')

        self.assertTrue(np.array_equal(self.MTCOV.u_f,theta['u']))
        self.assertTrue(np.array_equal(self.MTCOV.v_f,theta['v']))
        self.assertTrue(np.array_equal(self.MTCOV.w_f,theta['w']))
        self.assertTrue(np.array_equal(self.MTCOV.beta_f,theta['beta']))

        self.assertTrue(np.array_equal(thetaGT['u'],theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'],theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'],theta['w']))
        self.assertTrue(np.array_equal(thetaGT['beta'],theta['beta']))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()