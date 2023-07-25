import os
import numpy as np
import cupy as cp

def calc_QK(adjacency_matrices, max_vertices, dir_name, is_BH, use_d):
    os.makedirs(dir_name, exist_ok=True)

    graphs = {}

    for i, i_data in enumerate(adjacency_matrices):
        text = ''
        if i_data.shape[1] > max_vertices:
            continue

        graphs[i] = Graph.make_graph(i_data, use_d)
        save_graph_prob(dir_name, graphs[i], i)

        for j, j_data in enumerate(adjacency_matrices):
            if j_data.shape[1] > max_vertices:
                continue
            if i > j:
                continue

            graphs[j] = Graph.make_graph(j_data, use_d)
            save_graph_prob(dir_name, graphs[j], j)

            print(f'i: {i}, j: {j}')

            if is_BH:
                kernel_value = Graph.inner_product_BH(graphs[i], graphs[j])
            else:
                kernel_value = Graph.inner_product_SH(graphs[i], graphs[j])

            print(f'{kernel_value}')
            text += f'{i}-{j}\n{kernel_value}\n'
        with open(f'{dir_name}/{i}.pack', 'w') as f:
            f.write(text)

def save_graph_prob(dir_name, g, index):
    with open(f'{dir_name}/{index}.graph_prob', 'w') as f:
        f.write(f'{g.prob_of_measuring_0ket}')
    print(f'prob of graph:{index} is saved.')

class Graph:
    def __init__(self, adjacency_mat, use_d):
        self.adjacency_mat = adjacency_mat
        self.indices = cp.array([i for i in range(2 ** len(adjacency_mat))])
        self.use_d = use_d

    def used_indices(self, index):
        """e.g. when self.indices:=[0,1,2,3,4,5,6,7] and index:=2, return [0,0,0,0,1,1,1,1]"""
        return (self.indices & (2 ** index)) // (2 ** index)

    def encode(self):
        self.num_vertices = cp.zeros(self.indices.shape, dtype=np.int64)
        for i in range(len(self.adjacency_mat)):
            self.num_vertices += self.used_indices(i)

        self.num_edges = cp.zeros(self.indices.shape, dtype=np.int64)
        for row_index in range(self.adjacency_mat.shape[0]):
            for col_index in range(row_index + 1, self.adjacency_mat.shape[0]):
                if self.adjacency_mat[row_index, col_index] == 1:
                    self.num_edges += self.used_indices(row_index) * self.used_indices(col_index)

        if not self.use_d:
            return

        self.num_id1 = cp.zeros(self.indices.shape, dtype=np.int64)
        self.num_id2 = cp.zeros(self.indices.shape, dtype=np.int64)
        self.num_id3 = cp.zeros(self.indices.shape, dtype=np.int64)
        for row_index in range(self.adjacency_mat.shape[0]):
            ancilla_for_id = cp.zeros(self.indices.shape)
            for col_index in range(self.adjacency_mat.shape[0]):
                if self.adjacency_mat[row_index, col_index] == 1:
                    ancilla_for_id += self.used_indices(row_index) * self.used_indices(col_index)
            self.num_id1 += ancilla_for_id == 1
            self.num_id2 += ancilla_for_id == 2
            self.num_id3 += ancilla_for_id == 3
        del ancilla_for_id

    def remove_indices(self):
        """make feature vector `self.features`"""
        # 0<=v,d1,d2,d3<=28, 0<=e<=28*28=784, so v,d1,d2,d3 has 10**2 spaces and e has 10**3 spaces.
        if not self.use_d:
            features = cp.vstack((self.num_vertices, self.num_edges))
            features = features[0] + features[1] * (10 ** 2) # ve
        else:
            features = cp.vstack((self.num_vertices, self.num_edges, self.num_id1, self.num_id2, self.num_id3))
            features = features[0] + features[1] * (10 ** 2) + features[2] * (10 ** (2 + 3)) + features[3] * (10 ** (2 + 3 + 2)) + features[4] * (10 ** (2 + 3 + 2 + 2)) # veid1id2id3

        self.features = cp.unique(features, return_counts=True)

        self.prob_of_measuring_0ket = cp.sum(self.features[1] ** 2) / (2 ** (2 * self.adjacency_mat.shape[0]))
        divide_value = np.sqrt(cp.sum(self.features[1] ** 2))
        self.normalized_features = (self.features[0], self.features[1] / divide_value)

        del self.indices
        del self.num_vertices
        del self.num_edges
        if self.use_d:
            del self.num_id1
            del self.num_id2
            del self.num_id3

        self.features = (cp.asnumpy(self.features[0]), cp.asnumpy(self.features[1]))
        self.normalized_features = (cp.asnumpy(self.normalized_features[0]), cp.asnumpy(self.normalized_features[1]))

    @classmethod
    def make_graph(cls, adjacency_mat, use_d):
        g = cls(adjacency_mat, use_d)
        g.encode()
        g.remove_indices()
        return g

    @staticmethod
    def inner_product_BH(g1, g2):
        result = 0
        i1 = 0
        i2 = 0
        FEATURE = 0
        VALUE = 1
        while True:
            if len(g1.normalized_features[FEATURE]) == i1 or len(g2.normalized_features[FEATURE]) == i2:
                break

            if g1.normalized_features[FEATURE][i1] == g2.normalized_features[FEATURE][i2]:
                result += g1.normalized_features[VALUE][i1] * g2.normalized_features[VALUE][i2]
                i1 += 1
                i2 += 1
            elif g1.normalized_features[FEATURE][i1] < g2.normalized_features[FEATURE][i2]:
                i1 += 1
            else: # g1.normalized_features[FEATURE][i1] > g2.normalized_features[FEATURE][i2]:
                i2 += 1
        return result

    @staticmethod
    def inner_product_SH(g1, g2):
        true_vecs = (np.array(g1.features[1]), np.array(g2.features[1]))
        coeff = 2/(np.sum(true_vecs[1])/np.sum(true_vecs[0])*np.sum(true_vecs[0]**2) + np.sum(true_vecs[0])/np.sum(true_vecs[1])*np.sum(true_vecs[1]**2))

        result = 0
        i1 = 0
        i2 = 0
        FEATURE = 0
        VALUE = 1
        while True:
            if len(g1.features[FEATURE]) == i1 or len(g2.features[FEATURE]) == i2:
                break

            if g1.features[FEATURE][i1] == g2.features[FEATURE][i2]:
                result += g1.features[VALUE][i1] * g2.features[VALUE][i2]
                i1 += 1
                i2 += 1
            elif g1.features[FEATURE][i1] < g2.features[FEATURE][i2]:
                i1 += 1
            else: # g1.features[FEATURE][i1] > g2.features[FEATURE][i2]:
                i2 += 1

        return coeff * result
