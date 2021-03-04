from enum import Enum, auto
import numpy as np
import networkx as nx
from docopt import docopt
from grakel.datasets import fetch_dataset
from _calc_RW import calc_RW
from _calc_SP import calc_SP
from _calc_GS import calc_GS
from _calc_QK import calc_QK

MAX_VERTICES = 28

class Kernel(Enum):
    QK_BH_ve = auto()
    QK_BH_ved = auto()
    QK_SH_ve = auto()
    QK_SH_ved = auto()
    RW = auto()
    GS = auto()
    SP = auto()

__doc__ = '''Calculation of each kernel for given dataset. <dataset> list is written in README.md. <kernel> is selected from [{kernels}].
Usage:
  {f} <dataset> <kernel>
  {f} -h | --help
Options:
  -h --help  Show this screen.
'''.format(f=__file__, kernels=','.join([kernel.name for kernel in Kernel]))


def fetch_dataset_as_mat(dataset_name: str):
    mutag = fetch_dataset(dataset_name)
    mutag_data = [np.array(list(each_data[0])) for each_data in mutag.data]
    def make_graph(edges):
        g = nx.Graph()
        g.add_edges_from(edges)
        return g
    return [np.asarray(nx.to_numpy_matrix(make_graph(each_data))) for each_data in mutag_data]


def main():
    args = docopt(__doc__)

    kernel = Kernel[args['<kernel>']]
    dataset_name = args['<dataset>']

    adjacency_matrices = fetch_dataset_as_mat(dataset_name)
    if kernel == Kernel.RW:
        calc_RW(adjacency_matrices, MAX_VERTICES, f'random_walk-{dataset_name}')
    elif kernel == Kernel.SP:
        calc_SP(adjacency_matrices, MAX_VERTICES, f'shortest_path-{dataset_name}')
    elif kernel == Kernel.GS:
        calc_GS(adjacency_matrices, MAX_VERTICES, f'graphlet_sampling-{dataset_name}')
    elif kernel == Kernel.QK_BH_ve:
        calc_QK(adjacency_matrices, MAX_VERTICES, f'quantum_BH_ve-{dataset_name}', is_BH=True, use_d=False)
    elif kernel == Kernel.QK_BH_ved:
        calc_QK(adjacency_matrices, MAX_VERTICES, f'quantum_BH_ved-{dataset_name}', is_BH=True, use_d=True)
    elif kernel == Kernel.QK_SH_ve:
        calc_QK(adjacency_matrices, MAX_VERTICES, f'quantum_SH_ve-{dataset_name}', is_BH=False, use_d=False)
    elif kernel == Kernel.QK_SH_ved:
        calc_QK(adjacency_matrices, MAX_VERTICES, f'quantum_SH_ved-{dataset_name}', is_BH=False, use_d=True)

if __name__ == '__main__':
    main()
