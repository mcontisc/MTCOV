import unittest
import numpy as np
import yaml
import tools as tl
import cv_functions as cvfun


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    C = 2
    gamma = 0.5
    in_folder = '../data/input/'
    adj_name = 'adj_cv.csv'
    cov_name = 'X_cv.csv'
    ego = 'source'
    alter = 'target'
    egoX = 'Name'
    attr_name = 'Metadata'
    undirected = True
    flag_conv = 'log'
    batch_size = None
    cv_type = 'kfold'
    NFold = 5
    out_mask = False

    with open('setting_MTCOV.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    prng = np.random.RandomState(seed=conf['rseed'])
    rseed = prng.randint(1000)

    dataset = adj_name.split('.')[0]

    '''
    Import data
    '''
    A, B, X, nodes = tl.import_data(in_folder, adj_name=adj_name, cov_name=cov_name, ego=ego,
                                    alter=alter, egoX=egoX, attr_name=attr_name,
                                    undirected=undirected, force_dense=True)
    Xs = np.array(X)

    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        L = self.B.shape[0]
        N = self.B.shape[1]
        assert N == self.X.shape[0]

        '''
        Extract masks
        '''
        if self.cv_type == 'kfold':
            idxG = cvfun.shuffle_indicesG(N, L, rseed=self.rseed)
            idxX = cvfun.shuffle_indicesX(N, rseed=self.rseed)
        else:
            idxG = None
            idxX = None

        for fold in range(self.NFold):

            self.rseed += self.prng.randint(500)
            maskG, maskX = cvfun.extract_masks(N, L, idxG=idxG, idxX=idxX, cv_type=self.cv_type, NFold=self.NFold,
                                               fold=fold, rseed=self.rseed, out_mask=self.out_mask)

            '''
            Set up training dataset    
            '''
            B_train = self.B.copy()
            print(B_train.shape, maskG.shape)
            B_train[maskG > 0] = 0

            X_train = self.Xs.copy()
            X_train[maskX > 0] = 0

            self.conf['end_file'] = 'CV' + str(fold) + 'C' + str(self.C) + 'g' + str(self.gamma) + '_' + self.dataset

            U, V, W, BETA, logL = cvfun.train_running_model(B_cv=B_train, X_cv=X_train, flag_conv=self.flag_conv,
                                                            C=self.C, Z=self.X.shape[1], gamma=self.gamma,
                                                            undirected=self.undirected,
                                                            nodes=self.nodes, batch_size=self.batch_size, **self.conf)

            '''
            Load parameters
            '''
            outinference = self.conf['out_folder'] + 'thetaCV' + str(fold) + 'C' + str(self.C) + \
                           'g' + str(self.gamma) + '_' + self.dataset
            theta = np.load(outinference+'.npz')
            thetaGT = np.load('../data/output/5-fold_cv/thetaGT'+str(fold)+'C' + str(self.C) + \
                           'g' + str(self.gamma) + '_' + self.dataset + '.npz')

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