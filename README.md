# MTCOV community detection
Python implementation of the MTCOV algorithm described in:

- Community detection with node attributes in multilayer networks.

This is a new generative algorithm that incorporates both the topology of interactions and node attributes to extract overlapping communities in directed and undirected multilayer networks. 

Copyright (c) 2020 [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included
- `code` : Contains the Python implementation of the MTCOV algorithm, the code for performing the cross-validation procedure and the code for generating synthetic data.
- `data/input` : Contains an example of directed multilayer network, including the adjacency tensor and the design matrix. These are synthetic data.
- `data/output` : Contains some results for testing the code.

## Requirements
The project have been developed using Python 3.7 with the packages contained in _requirements.txt_. It is possible to install the dependencies using pip:
`pip install -r requirements.txt`

## Test
You can run tests to reproduce results contained in `data/output` by running (inside `code` directory):  
`python -m unittest test.py`   
`python -m unittest test_cv.py`   
## Usage
To test the program on the given example file, type:  
`cd code;`   
`python main.py`

It will use the sample network contained in `./data/input`. The adjacency tensor _adj.csv_ represents a directed, unweighted network with **N=300** nodes and **L=4** layers. The design matrix _X.csv_ contains the attribute _Metadata_ with **Z=2** modalities used in the analysis. The algorithm runs with **C=2** communities and **gamma=0.5**. 

### Parameters
- **-j** : Input file name of the adjacency tensor, *(default='adj.csv')*.
- **-c** : Input file name of the design matrix, *(default='X.csv')*.
- **-o** : Name of the source of the edge, *(default='source')*.
- **-r** : Name of the target of the edge, *(default='target')*.
- **-a** : Name of the attribute to consider in the analysis, *(default='Metadata')*.
- **-I** : Name of the column with node labels, *(default='Name')*.
- **-C** : Number of communities, *(default=2)*.
- **-g** : Scaling parameter gamma, *(default=0.5)*.
- **-u** : Flag to call the undirected network, *(default=False)*.
- **-d** : Flag to force a dense transformation of the adjacency tensor, *(default=False)*.
- **-z** : Seed for random real numbers, *(default=107261)*.
- **-e** : Error for the initialization of W, *(default=0.1)*.
- **-i** : Number of iterations with different random initialization; the final parameters will be the one corresponding to the realization leading to the max likelihood, *(default=1)*.
- **-t** : Tolerance parameter for convergence, *(default=0.1)*.
- **-y** : Decision variable for convergence, *(default=10)*.
- **-m** : Maximum number of EM steps before aborting, *(default=500)*.
- **-E** : Output file suffix, *(default='.dat')*.

## Input format
The multilayer network should be stored in a CSV file. An example of row is

`node1 node2 0 0 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively; l+2 column tells if there is that edge in the l-th layer and the weight. In this example the edge node1 --> node2 exists in layer 3 with weight 1, but not in layer 1 and 2.

Note: if the network is undirected, you only need to input each edge once. You then need to specify to the algorithm that you are considering the undirected case by giving as a command line input parameter **-u=True**. 

The design matrix should be stored in a CSV file, where one column (_Name_) indicates the node labels. 

## Output
The MTCOV returns four files inside the `data/output` folder: the two NxC membership matrices **U** and **V**, the CxCxL affinity tensor **W** and the CxZ matrix **beta**. 

The first line outputs the maximum log-likelihood (*Max Likelihood*) among the different iterations performed with different random initialization (*NReal*) and the value of the *gamma* scaling parameter.

For the membership files, the subsequent lines contain C+1 columns: the first one is the node label, the following ones are the membership probabilities.

For the affinity tensor file, the subsequent lines start with the number of the layer and then the affinity matrix for that layer.

For the beta file, the subsequent line contains the names of the modalities of the categorical attribute. Then, the following lines contain Z columns with the probabilities indicating the correlation between communities and attributes.

