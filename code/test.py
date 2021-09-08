import unittest
import numpy as np
import yaml
import MTCOV as mtcov
import tools as tl


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
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
    undirected = False
    flag_conv = 'log'
    force_dense = False
    batch_size = None

    with open('setting_MTCOV.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    '''
    Import data
    '''
    A, B, X, nodes = tl.import_data(in_folder, adj_name=adj_name, cov_name=cov_name, ego=ego,
                                    alter=alter, egoX=egoX, attr_name=attr_name,
                                    undirected=undirected, force_dense=force_dense)
    Xs = np.array(X)

    MTCOV = mtcov.MTCOV(N=A[0].number_of_nodes(), L=len(A), C=C, Z=X.shape[1], gamma=gamma,
                        undirected=undirected, **conf)

    def test_import_data(self):
        print("Start import data test\n")
        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
            print('B has ', self.B.sum(), ' total weight.')
        else:
            self.assertTrue(self.B.vals.sum() > 0)
            print('B has ', self.B.vals.sum(), ' total weight.')

    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        _ = self.MTCOV.fit(data=self.B, data_X=self.Xs, flag_conv=self.flag_conv,
                           nodes=self.nodes, batch_size=self.batch_size)

        theta = np.load(self.MTCOV.out_folder+'theta'+self.MTCOV.end_file+'.npz')
        thetaGT = np.load(self.MTCOV.out_folder+'theta_test_GT.npz')

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