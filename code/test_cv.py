
import unittest
import numpy as np
import pandas as pd

import MTCOV as mtcov
import tools as tl
import cv_functions as cvfun


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
    end_file = '_test_cv'
    adj_name = 'adj_cv.csv'
    cov_name = 'X_cv.csv'
    ego = 'source'
    alter = 'target'
    egoX = 'Name'
    attr_name = 'Metadata'
    rseed = 107261
    N_real = 1

    cv_type = 'random'
    NFold = 5

    '''
    Import data
    '''
    A, B, X, u_list, v_list = tl.import_data(in_folder, adj_name=adj_name, cov_name=cov_name, ego=ego,
                                             alter=alter, egoX=egoX, attr_name=attr_name)

    # test case function to check the Person.get_name function
    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        L = self.B.shape[0]
        N = self.B.shape[1]
        assert N == self.X.shape[0]

        if self.cv_type == 'kfold':
            idxG = cvfun.shuffle_indicesG(N,L,rseed=rseed)
            idxX = cvfun.shuffle_indicesX(N,rseed=rseed)
        else:
            idxG = None
            idxX = None

        for fold in range(self.NFold):

            ind = self.rseed+fold

            maskG, maskX = cvfun.extract_masks(self.N,self.L,idxG=idxG, idxX=idxX, cv_type=self.cv_type, NFold=self.NFold, fold=fold, rseed=ind)

            '''
            Set up training dataset    
            '''            
            B_train = self.B.copy()
            B_train[maskG > 0] = 0 

            X_train = self.X.copy()
            X_train.iloc[maskX > 0] = 0

            U, V, W, BETA, logL = cvfun.train_running_model(self.A, B_train, X_train, self.u_list, self.v_list,
                                                                         N=self.A[0].number_of_nodes(),
                                                                         L=len(self.B),
                                                                         C=self.C,
                                                                         Z=self.X.shape[1],
                                                                         gamma=self.gamma,
                                                                         cv=True,
                                                                         rseed=self.rseed,
                                                                         N_real=self.N_real,
                                                                         folder=self.out_folder,
                                                                         end_file=self.end_file)
            '''
            Output parameters
            '''
            outinference = self.out_folder+'theta_cv'+str(fold)+'C'+str(self.C)+'g'+str(self.gamma)
            np.savez_compressed(outinference+'.npz', u=U, v=V, w=W, beta=BETA, fold=fold) # To load: theta = np.load('test.npz'), e.g. print(np.array_equal(U, theta['u']))
            
            '''
            Load parameters
            '''            
            theta = np.load(outinference+'.npz')
            thetaGT = np.load(self.out_folder+'thetaGT'+str(fold)+'C2g0.5_adj_cv.npz')

            self.assertTrue(np.array_equal(U,theta['u']))
            self.assertTrue(np.array_equal(V,theta['v']))
            self.assertTrue(np.array_equal(W,theta['w']))
            self.assertTrue(np.array_equal(BETA,theta['beta']))

            self.assertTrue(np.array_equal(thetaGT['u'],theta['u']))
            self.assertTrue(np.array_equal(thetaGT['v'],theta['v']))
            self.assertTrue(np.array_equal(thetaGT['w'],theta['w']))
            self.assertTrue(np.array_equal(thetaGT['beta'],theta['beta']))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()