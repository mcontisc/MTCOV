# MTCOV community detection
Python implementation of MTCOV algorithm described in:

- [1] Contisciani M., Power E. & De Bacco C. (2020). *Community detection with node attributes in multilayer networks*,  Scientific Reports **10**, 15736 (2020).  

This is a new generative algorithm that incorporates both the topology of interactions and node attributes to extract overlapping communities in directed and undirected multilayer networks. 

If you use this code please cite [1].   

The paper can be found [here](https://www.nature.com/articles/s41598-020-72626-y) (_Published version, open access_) or [here](https://arxiv.org/abs/2004.09160) (_preprint_).  

Copyright (c) 2020 [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included
- `code` : Contains the Python implementation of MTCOV algorithm, the code for performing the cross-validation procedure and the code for generating synthetic data.
- `data/input` : Contains an example of directed multilayer network, including the adjacency tensor and the design matrix. These are synthetic data.
- `data/output` : Contains some results for testing the code.

## Requirements
The project has been developed using Python 3.7 with the packages contained in _requirements.txt_. It is possible to install the dependencies using pip:
`pip install -r requirements.txt`

## Test
You can run tests to reproduce results contained in `data/output` by running (inside `code` directory):  
`python -m unittest test.py`   
`python -m unittest test_cv.py`   
## Usage
To test the program on the given example file, type:  
`cd code`   
`python main.py`

It will use the sample network contained in `./data/input`. The adjacency tensor _adj.csv_ represents a directed, unweighted network with **N=300** nodes and **L=4** layers. The design matrix _X.csv_ contains the attribute _Metadata_ with **Z=2** modalities used in the analysis. The algorithm runs with **C=2** communities and **gamma=0.5**. 

### Parameters
- **-f** : Path of the input folder, *(default='../data/input/')*.
- **-j** : Input file name of the adjacency tensor, *(default='adj.csv')*.
- **-c** : Input file name of the design matrix, *(default='X.csv')*.
- **-o** : Name of the source of the edge, *(default='source')*.
- **-r** : Name of the target of the edge, *(default='target')*.
- **-x** : Name of the column with node labels, *(default='Name')*.
- **-a** : Name of the attribute to consider in the analysis, *(default='Metadata')*.
- **-C** : Number of communities, *(default=2)*.
- **-g** : Scaling parameter gamma, *(default=0.5)*.
- **-u** : Flag to call the undirected network, *(default=False)*.
- **-d** : Flag to force a dense transformation of the adjacency tensor, *(default=False)*.
- **-F** : Flag to choose the convergence procedure, *(default='log')*. If 'log' the convergence is based on the loglikelihood values; if 'deltas' the convergence is based on the differences in the parameters values. The latter is suggested when the dataset is big (N > 1000 ca.).
- **-b** : Size of the batch used to compute the likelihood, *(default=None)*.


## Input format
The multilayer network should be stored in a CSV file. An example of row is

`node1 node2 0 0 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively; l+2 column tells if there is that edge in the l-th layer and the weight. In this example the edge node1 --> node2 exists in layer 3 with weight 1, but not in layer 1 and 2.

Note: if the network is undirected, you only need to input each edge once. You then need to specify to the algorithm that you are considering the undirected case by giving as a command line input parameter **-u=True**. 

The design matrix should be stored in a CSV file, where one column (_Name_) indicates the node labels. 

## Output
The MTCOV returns a compressed file inside the `data/output/test` folder. To load and print the out-going membership matrix:

`theta = np.load('theta_test_GT.npz')`.      
`print(theta['u'])`

*theta* contains the two NxC membership matrices **U** *('u')* and **V** *('v')*, the LxCxC affinity tensor **W** *('w')*, the CxZ matrix **beta** *('beta')*, the total number of iterations *('max_it')*, the nodes of the network *('nodes')*, the value of the maximum likelihood *('maxL')* and the number of realizations *('N_real')*. 
