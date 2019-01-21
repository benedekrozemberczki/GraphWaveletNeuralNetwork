import json
import pygsp
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to create an NX graph object.
    :param path: Path to the edge list csv.
    :return graph: NetworkX graph.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    features = json.load(open(path))
    index_1 = [int(k) for k,v in features.items() for fet in v]
    index_2 = [int(fet) for k,v in features.items() for fet in v]
    values = [1.0]*len(index_1) 
    nodes = [int(k) for k,v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.csr_matrix(sparse.coo_matrix((values,(index_1,index_2)),shape=(node_count,feature_count),dtype=np.float32))
    return features

def target_reader(path):
    """
    Reading thetarget vector to a numpy column vector.
    :param path: Path to the target csv.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"])
    return target

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path,"w") as f:
        json.dump(logs,f)

class WaveletSparsifier(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """
    def __init__(self, graph, scale, approximation_order, tolerance):
        """
        :param graph: NetworkX graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.graph = graph
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.graph))
        self.pygsp_graph.estimate_lmax()
        self.scales = [scale, -scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self, node):
        """
        Creating sparse wavelets from a source node:
        :param node: Source node.
        :return remaining_waves: Dictionary of attenuated wavelets.
        """
        impulse = np.zeros((self.graph.number_of_nodes()))
        impulse[node] = 1.0
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph, self.chebyshev, impulse)
        remaining_waves = {target: wave for target, wave in enumerate(wavelet_coefficients) if wave > self.tolerance}
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        
        print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            index_1 = [k for k, v in phi_matrix.items() for ke, ve in v.items()]
            index_2 = [ke for k, v in phi_matrix.items() for ke, ve in v.items()]
            scores = [ve for k, v in phi_matrix.items() for ke, ve in v.items()]
            nodes = max(index_1)+1
            self.phi_matrices[i] = sparse.csr_matrix(sparse.coo_matrix((scores,(index_1,index_2)),shape=(nodes,nodes),dtype=np.float32))
            self.phi_matrices[i] = self.phi_matrices[i]/self.phi_matrices[i].sum(axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = str(round(100*len(self.phi_matrices[0].nonzero()[0])/(self.graph.number_of_nodes()**2),2))
        inverse_wavelet_density = str(round(100*len(self.phi_matrices[1].nonzero()[0])/(self.graph.number_of_nodes()**2),2))
        print("Density of wavelets: "+wavelet_density+"%.")
        print("Density of inverse wavelets: "+inverse_wavelet_density+"%.\n")

    def calculate_all_wavelets(self):
        """
        Graph wavelet coefficient calculation for each node.
        """
        print("\nWavelet calculation and sparsification started.\n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph, tau = [scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m = self.approximation_order)
            sparsified_wavelets = {node: self.calculate_wavelet(node) for node in tqdm(self.graph.nodes())}
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()
