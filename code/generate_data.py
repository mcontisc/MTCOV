""" Generate synthetic networks with communities. """

import os
import csv
import networkx as nx
import numpy as np
import tools as tl
from networkx.generators.community import stochastic_block_model
import matplotlib.pyplot as plt


def main_generate_data():
    N_graph = 1  # number of graphs

    for graph in range(N_graph):
        print('### GRAPH {0} ###'.format(graph))

        # stochastic block model
        C = 2  # number of communities
        N = 100  # number of nodes
        deltak = int(N / C)  # number of nodes per community (communities have equal size)
        L = 2  # number of layers
        L_a = 1  # number of assortative layers
        sizes = [deltak] * C  # sizes of blocks, here equal size

        if not os.path.exists('../data/input' + str(C) + str(L) + str(L_a) + '/graph' + str(graph)):
            os.makedirs('../data/input' + str(C) + str(L) + str(L_a) + '/graph' + str(graph))

        # generate the network structure
        G = [nx.MultiDiGraph() for _ in range(L)]
        for l in range(L):
            if l < L_a:
                p = tl.probabilities('assortative', sizes, N, C)
                G[l] = stochastic_block_model(sizes, p, directed=True)
                # Adj = nx.to_numpy_matrix(G[l], weight='weight')
                # plt.imshow(Adj, cmap="Greys", interpolation="nearest")
                # plt.show()
            elif l == 1:
                p = tl.probabilities('disassortative', sizes, N, C)
                G[l] = stochastic_block_model(sizes, p, directed=True)
                # Adj = nx.to_numpy_matrix(G[l], weight='weight')
                # plt.imshow(Adj, cmap="Greys", interpolation="nearest")
                # plt.show()
            elif l == 2:
                p = tl.probabilities('core-periphery', sizes, N, C)
                G[l] = stochastic_block_model(sizes, p, directed=True)
                # Adj = nx.to_numpy_matrix(G[l], weight='weight')
                # plt.imshow(Adj, cmap="Greys", interpolation="nearest")
                # plt.show()
            elif l == 3:
                p = tl.probabilities('directed-biased', sizes, N, C)
                G[l] = stochastic_block_model(sizes, p, directed=True)
                # Adj = nx.to_numpy_matrix(G[l], weight='weight')
                # plt.imshow(Adj, cmap="Greys", interpolation="nearest")
                # plt.show()

            print('Nodes: ', G[l].number_of_nodes())
            print('Edges: ', G[l].number_of_edges())

        # save the graph
        folder = '../data/input' + str(C) + str(L) + str(L_a) + '/graph' + str(graph) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        tl.write_adjacency(G, folder=folder, fname='adj.csv', ego='source', alter='target')

        # generate the covariates
        for perc in [0.3, 0.5, 0.7, 0.9]:  # loop over fractions of match between communities and metadata
            if (perc is not None) and (perc > 0.):
                metadata = {}
                nodes_match = np.random.choice(np.array(G[0].nodes()), size=int(N * perc), replace=False)
                for i in G[0].nodes():
                    if i in nodes_match:
                        metadata[i] = ('Meta0' if i < deltak else 'Meta1')
                    else:
                        metadata[i] = 'Meta' + str(np.random.randint(2, size=1)[0])
                # name_file = 'nodes_match_' + str(perc)[0] + '_' + str(perc)[2]
                # with open('../data/input' + str(C) + str(L) + str(L_a) + '/graph' + str(graph) + '/' + name_file,
                #           'w', newline='') as myfile:
                #     csv.writer(myfile, quoting=csv.QUOTE_ALL)

                tl.write_design_Matrix(metadata, perc, folder=folder, fname='X_', nodeID='Name', attr_name='Metadata')


if __name__ == '__main__':
    main_generate_data()
