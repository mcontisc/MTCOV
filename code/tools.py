""" Functions for handling the data. """

import pandas as pd
import networkx as nx
import numpy as np
from termcolor import colored

# print(nx.__version__)


def import_data(in_folder, adj_name='adj.csv', cov_name='X.csv', ego='source', egoX='Name',alter='target', attr_name='Metadata',
                 undirected=False, force_dense=True):
    """
        Import data, i.e. the adjacency tensor and the design matrix, from a given folder.

        Return the NetworkX graph, its numpy adjacency tensor and the dummy version of the design matrix.

        Parameters
        ----------
        in_folder : str
                    Path of the folder containing the input files.
        adj_name : str
                   Input file name of the adjacency tensor.
        cov_name : str
                   Input file name of the design matrix.
        ego : str
              Name of the column to consider as source of the edge.
        egoX : str
              Name of the column to consider in the design matrix-attribute dataset.
        alter : str
                Name of the column to consider as target of the edge.
        attr_name : str
                    Name of the attribute to consider in the analysis.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.
        force_dense : bool
                      If set to True, the algorithm is forced to consider a dense adjacency tensor. Default is True because current implementation is not suited to manage sparse tensors

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
        B : ndarray
            Graph adjacency tensor.
        X_attr : DataFrame
                Pandas DataFrame object representing the one-hot encoding version of the design matrix.
        u_list : list
                 List of indexes of nodes with non zero out-degree over all layers.
        v_list : list
                 List of indexes of nodes with non zero in-degree over all layers.
    """

    # df_adj = pd.read_csv(in_folder + adj_name, index_col=0) # read adjacency file
    df_adj = pd.read_csv(in_folder + adj_name) # read adjacency file
    print('\nAdjacency shape: {0}'.format(df_adj.shape))

    df_X = pd.read_csv(in_folder + cov_name) # read the csv file with the covariates
    print('Indiv shape: ', df_X.shape)
    
    A = read_graph(df_adj=df_adj, ego=ego, alter=alter, undirected=undirected) # create the graph adding nodes and edges
    print_graph_stat(A)

    N = A[0].number_of_nodes()
    nodes = list(A[0].nodes)
    
    B = build_B_from_A(A,force_dense=force_dense, nodes=nodes) # save the multilayer network in a numpy tensor with all layers

    # save indexes of nodes with non zero in/out degree
    if undirected:
        u_list = v_list = remove_zero_entries_undirected(A)
    else:
        u_list = remove_zero_entries_u(A)
        v_list = remove_zero_entries_v(A)

    # read the design matrix with covariates
    X_attr=read_design_matrix(df_X, nodes,attribute=attr_name, ego=egoX)

    return A, B, X_attr, u_list, v_list


def read_graph(df_adj, ego='source', alter='target', undirected=False):
    """
        Create the graph by adding edges and nodes.

        Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

        Parameters
        ----------
        adj : DataFrame
              Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        undirected : bool
                     If set to True, the algorithm considers an undirected graph.

        Returns
        -------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L=df_adj.shape[1] - 2 # number of layers
    # build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers.
    if undirected:
        A = [nx.MultiGraph() for _ in range(L)]
    else:
        A = [nx.MultiDiGraph() for _ in range(L)]

    print('Creating the network ...', end=' ')
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2]>0:
                if A[l].has_edge(v1,v2): A[l][v1][v2][0]['weight'] += int(row[l + 2]) # if the edge existed already, no parallel edges created
                else: A[l].add_edge(v1, v2, weight=int(row[l + 2]))
    print('done!')

    # remove self-loops
    for l in range(L):
        A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A

def build_B_from_A(A,force_dense=True, nodes=None):
    N = A[0].number_of_nodes()
    if nodes is None: nodes=list(A[0].nodes())
    B = np.empty(shape=[len(A), N, N])
    for l in range(len(A)):
        if force_dense==True:
            B[l,:,:]=nx.to_numpy_matrix(A[l],weight='weight',dtype=int,nodelist=nodes)
        else:
            try:
                B[l,:,:]=nx.to_scipy_sparse_matrix(A[l],dtype=int,nodelist=nodes) #scipy.sparse.csr_matrix(B[l,:,:])
            except:
                print('Warning: layer ',l, ' cannot be made sparse, using a dense matrix for it')
                B[l,:,:]=nx.to_numpy_matrix(A[l],weight='weight',dtype=int,nodelist=nodes)
    return B

def read_design_matrix(df_X, nodes,attribute=None,ego='Name'):
    X = df_X[df_X[ego].isin(nodes)] # filter nodes 
    X = X.set_index(ego).loc[nodes].reset_index() # sort by nodes

    if attribute is None:X_attr = pd.get_dummies(X.iloc[:,1]) # gets the first columns after the ego
    else:
        if 'CasteAge' in attribute: # use two attributes and one is binned#
            print("Using CasteAGe")
            ageyear = attribute.split('Caste')[-1] # extract the age_year from e.g. 'CasteAge_2013'
            bins = [X[ageyear].min()-1+i*5 for i in range(20) if X[ageyear].min()-1+i*5-5<=X[ageyear].max() ]
            labels=['b'+str(i) for i in range(len(bins)-1)]
            X.loc[:,ageyear+'_bin'] = pd.cut(X[ageyear],bins=bins,labels=labels)
            d=[ r['Caste']+'_'+r[ageyear+'_bin'] for i,r in X.iterrows()]
            X.loc[:,attribute] = d
        elif 'Age' in attribute: # use one binned attribute
            bins = [X[attribute].min()-1+i*5 for i in range(20) if X[attribute].min()-1+i*5-5<=X[attribute].max() ]
            labels=['b'+str(i) for i in range(len(bins)-1)]
            X_attr = pd.cut(X[attribute],bins=bins,labels=labels)
            X_attr = pd.get_dummies(X_attr)
            assert X_attr.sum(axis=1).sum()==len(X)
        else: # use one attribute as it is (not binned)
            X_attr = pd.get_dummies(X[attribute])
    print('\nDesign matrix shape: ', X_attr.shape)

    print('Distribution of attribute {0}: '.format(attribute))
    print(np.sum(X_attr, axis=0))
    return X_attr

def print_graph_stat(A):
    """
        Print the statistics of the graph A: number of nodes, number of layers, number of edges and average degree in
        each layer, average density, average and total number of edges.

        Parameters
        ----------
        A : list
            List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
    """

    L = len(A)  # number of layers
    N = A[0].number_of_nodes()
    avg_edges = 0
    avg_degrees = 0
    print('Number of nodes =', N)
    print('Number of layers =', L)
    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = A[l].number_of_edges()
        k = 2 * float(E) / float(N)
        # density = 100 * float(E) / float(N * (N - 1))
        print('E[', l, '] =', E, " -  <k> =", np.round(k, 2))
        avg_edges += E
        avg_degrees += k
    print('Average degree over all layers:', np.round(avg_degrees / L, 2))
    print('Average edges over all layers:', np.round(avg_edges / L, 3))
    print('Total number of edges:', avg_edges)


def can_cast(string):
    """
        Verify if one object can be converted to integer object.

        Parameters
        ----------
        string : int or float or str
                 Name of the node.

        Returns
        -------
        bool : bool
               If True, the input can be converted to integer object.
    """

    try:
        int(string)
        return True
    except ValueError:
        return False


def remove_zero_entries_undirected(A):
    """
        Return the indexes of the nodes with non zero degree, i.e. the non isolated vertices.

        Parameters
        ----------
        A : list
            List of MultiGraph NetworkX objects.

        Returns
        -------
        zero : list
               List of indexes of nodes having non zero degree over the all layers.
    """

    L = len(A)  # number of layers
    zero = []
    nodes = list(A[0].nodes())
    for i in nodes:
        k = 0
        for l in range(L):
            if type(A[l].degree(i)) != dict:
                k += A[l].degree(i)
        if k > 0:
            zero.append(nodes.index(i))
    return zero


def remove_zero_entries_u(A):
    """
        Return the indexes of the nodes with non zero out-degree.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        zero : list
               List of indexes of nodes having non zero out-degree over the all layers.
    """

    L = len(A)  # number of layers
    zero_out = []
    nodes = list(A[0].nodes())
    for i in nodes:
        k = 0
        for l in range(L):
            if type(A[l].out_degree(i)) != dict:
                k += A[l].out_degree(i)
        if k > 0:
            zero_out.append(nodes.index(i))
    return zero_out


def remove_zero_entries_v(A):
    """
        Return the indexes of the nodes with non zero in-degree.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        zero : list
               List of indexes of nodes having non zero in-degree over the all layers.
    """

    L = len(A)  # number of layers
    zero_in = []
    nodes = list(A[0].nodes())
    for i in nodes:  # cycle over nodes
        k = 0
        for l in range(L):
            if type(A[l].in_degree(i)) != dict:
                k += A[l].in_degree(i)
        if k > 0:
            zero_in.append(nodes.index(i))
    return zero_in


def probabilities(structure, sizes, N=100, C=2, avg_degree=4.):
    """
        Return the CxC array with probabilities between and within groups.

        Parameters
        ----------
        structure : str
                    Structure of the layer, e.g. assortative, disassortative, core-periphery or directed-biased.
        sizes : list
                List with the sizes of blocks.
        N : int
            Number of nodes.
        C : int
            Number of communities.
        avg_degree : float
                     Average degree over the nodes.

        Returns
        -------
        p : ndarray
            Array with probabilities between and within groups.
    """

    alpha = 0.1
    beta = alpha * 0.3
    p1 = avg_degree * C / N
    if structure == 'assortative':
        p = p1 * alpha * np.ones((len(sizes), len(sizes)))  # primary-probabilities
        np.fill_diagonal(p, p1)  # secondary-probabilities
    elif structure == 'disassortative':
        p = p1 * np.ones((len(sizes), len(sizes)))
        np.fill_diagonal(p, alpha * p1)
    elif structure == 'core-periphery':
        p = p1 * np.ones((len(sizes), len(sizes)))  # primary-probabilities
        np.fill_diagonal(np.fliplr(p), alpha * p1)  # secondary-probabilities
        p[1, 1] = beta * p1
    elif structure == 'directed-biased':
        p = alpha * p1 * np.ones((len(sizes), len(sizes)))
        p[0, 1] = p1
        p[1, 0] = beta * p1

    print(p)
    return p


def write_adjacency(G, folder='./', fname='adj.csv', ego='source', alter='target'):
    """
        Save the adjacency tensor to file.

        Parameters
        ----------
        G : list
            List of MultiDiGraph NetworkX objects.
        folder : str
                 Path of the folder where to save the files.
        fname : str
                Name of the adjacency tensor file.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
    """

    N = G[0].number_of_nodes()
    L = len(G)
    B = np.empty(shape=[len(G), N, N])
    for l in range(len(G)):
        B[l, :, :] = nx.to_numpy_matrix(G[l], weight='weight')
    df = []
    for i in range(N):
        for j in range(N):
            Z = 0
            for l in range(L):
                Z += B[l][i][j]
            if Z > 0:
                data = [i, j]
                data.extend([int(B[a][i][j]) for a in range(L)])
                df.append(data)
    cols = [ego, alter]
    cols.extend(['L' + str(l) for l in range(1, L + 1)])
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(path_or_buf=folder + fname,index=False)
    print('Adjacency tensor saved in:', folder + fname)


def write_design_Matrix(metadata, perc, folder='./', fname='X_', nodeID='Name', attr_name='Metadata'):
    """
        Save the design matrix to file.

        Parameters
        ----------
        metadata : dict
                   Dictionary where the keys are the node labels and the values are the metadata associated to them.
        perc : float
               Fraction of match between communities and metadata.
        folder : str
                 Path of the folder where to save the files.
        fname : str
                Name of the design matrix file.
        nodeID : str
                 Name of the column with the node labels.
        attr_name : str
                    Name of the column to consider as attribute.
    """

    X = pd.DataFrame.from_dict(metadata, orient='index', columns=[attr_name])
    X[nodeID] = X.index
    X = X.loc[:, [nodeID, attr_name]]
    X.to_csv(path_or_buf=folder + fname + str(perc)[0] + '_' + str(perc)[2] + '.csv',index=False)
    print('Design matrix saved in:', folder + fname + str(perc)[0] + '_' + str(perc)[2] + '.csv')
