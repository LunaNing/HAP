# -*- coding: utf-8 -*-

from config import FLAGS
from data import Data
from utils_model import load_data
from sklearn.preprocessing import OneHotEncoder
# from utils_model import get_coarsen_level
# from coarsening import coarsen,perm_data
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


class ModelData(Data):
    def __init__(self):
        self.dataset = FLAGS.dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_name = FLAGS.node_feat_name
        self.node_feat_encoder = FLAGS.node_feat_encoder
        # self.bsf_ordering = FLAGS.bfs_ordering
        # self.coarsening = FLAGS.coarsening
        # self.random_permute = FLAGS.random_permute
        super().__init__(self._get_name())
        print('{} train graphs; {} validation graphs; {} test graphs'.format(
            len(self.train_graphs), len(self.val_graphs), len(self.test_graphs)))

    def init(self):
        orig_data = load_data(self.dataset, train=True)
        train_graphs, valid_graphs = self._train_val_split(orig_data)
        test_graphs = load_data(self.dataset, train=False).graphs
        self.node_feat_encoder = self._get_node_feature_encoder(orig_data.graphs + test_graphs)
        self._check_graph_num(test_graphs, 'test')
        self.train_graphs = [ModelGraph(g, self.node_feat_encoder) for g in train_graphs]
        self.val_graphs = [ModelGraph(g, self.node_feat_encoder) for g in valid_graphs]
        self.test_graphs = [ModelGraph(g, self.node_feat_encoder) for g in test_graphs]
        assert (len(train_graphs) + len(valid_graphs) == len(orig_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def _get_name(self):
        item = []
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            item.append('{}'.format(v))
        return '_'.join(item)



    def _train_val_split(self, orig_data):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0,1]'.format(self.valid_percentage))
        gs = orig_data.graphs
        split = int(len(gs) * (1 - self.valid_percentage))
        train_graphs = gs[0:split]
        valid_graphs = gs[split:]
        self._check_graph_num(train_graphs, 'train')
        self._check_graph_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    def _get_node_feature_encoder(self, gs):
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        elif 'constant' in self.node_feat_encoder:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(self.node_feat_encoder))

    @staticmethod
    def _check_graph_num(graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(label, len(graphs)))


class NodeFeatureEncoder(object):
    def encode(self, g):
        raise NotImplementedError()

    def input_dim(self):
        raise NotImplementedError()


class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())  # the union of two sets
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self.oe = OneHotEncoder().fit(
            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))  # transform to one column

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        rtn = np.full((g.number_of_nodes(), self.input_dim_),
                      self.const)  # g is a nx graph, number_of_nodes is a method in nx
        return rtn

    def input_dim(self):
        return self.input_dim_


class ModelGraph(object):
    def __init__(self, nxgraph, node_feat_encoder):
        self.nxgraph = nxgraph
        self.dense_node_inputs = node_feat_encoder.encode(nxgraph)
        # if FLAGS.random_permute:
        #     self.graph_size = len(nxgraph.nodes())
        #     self.permute_order = np.random.permutation(self.graph_size)
        self.sparse_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(self.dense_node_inputs))
        self.adj = np.nan_to_num(nx.to_numpy_matrix(nxgraph))
        self.num_laplacians = 1
        self.laplacians = [self._preprocess_adj(self.adj)]
        # if FLAGS.coarsening:
        #     self._coarsen()

    def get_nxgraph(self):
        return self.nxgraph

    def get_adj_matrix(self):
        return self.adj

    def get_laplacians(self, gcn_id):
       return self.laplacians

    def get_node_inputs(self):
        # if FLAGS.coarsening:
        #     return self.sparse_permuted_padded_dense_node_inputs
        # else:
        return self.sparse_node_inputs

    def get_node_inputs_num_nonzero(self):
        return self.get_node_inputs()[1].shape

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return self._sparse_to_tuple(inputs)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # def _coarsen(self):
    #     assert ('metis_' in FLAGS.coarsening)
    #     self.num_level = get_coarsen_level()
    #     assert (self.num_level >= 1)
    #     graphs, perm = coarsen(sp.csr_matrix(self.adj), levels=self.num_level,
    #                            self_connections=False)
    #     self.permuted_padded_dense_node_inputs = perm_data(
    #         self.dense_node_inputs.T, perm).T
    #     self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(
    #         sp.csr_matrix(self.permuted_padded_dense_node_inputs))

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


class S2VGraph(object):
    def __init__(self, g, label, adj_matrix, node_features, node_tags=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.adj_matrix = adj_matrix
        self.edge_mat = 0
        self.laplacians =[self._preprocess_adj(self.adj_matrix)]

        self.max_neighbor = 0

    def get_node_features(self):
        return self.node_features

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_laplacians(self, gcn_id):
       return self.laplacians

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

        # def _coarsen(self):
        #     assert ('metis_' in FLAGS.coarsening)
        #     self.num_level = get_coarsen_level()
        #     assert (self.num_level >= 1)
        #     graphs, perm = coarsen(sp.csr_matrix(self.adj), levels=self.num_level,
        #                            self_connections=False)
        #     self.permuted_padded_dense_node_inputs = perm_data(
        #         self.dense_node_inputs.T, perm).T
        #     self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(
        #         sp.csr_matrix(self.permuted_padded_dense_node_inputs))

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


def load_classification_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('/home/ln/spyder-workspace/HAP-master-for-graph-classification/data/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            adj_matrix = np.nan_to_num(nx.to_numpy_matrix(g))

            g_list.append(S2VGraph(g, l, adj_matrix, node_features, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]
        # print('g.label:', g.label)


        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = tf.transpose(tf.convert_to_tensor(edges, name='edges', dtype=tf.int64))


    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree()).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros([len(g.node_tags), len(tagset)])
        # g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        # g.node_features = tf.convert_to_tensor(g.node_features, name='node_features', dtype=tf.float32)

    max_nodes = max([len(G.node_tags) for G in g_list])
    mean_nodes = int(np.mean([len(G.node_tags) for G in g_list]))

    print('#max nodes: %d' % max_nodes)
    print('#mean nodes: %d' % mean_nodes)

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

# def separate_data(graph_list, seed, fold_idx):
#     assert 1 <= fold_idx and fold_idx <= 10, "fold_idx must be from 1 to 10."
#     skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
#
#     labels = [graph.label for graph in graph_list]
#     idx_list = []
#     for idx in skf.split(np.zeros(len(labels)), labels):
#         idx_list.append(idx)
#     train_idx, test_idx = idx_list[fold_idx]
#
#     train_graph_list = [graph_list[i] for i in train_idx]
#     # test_graph_list = [graph_list[i] for i in test_idx]
#     test_graph_list = graph_list
#     return train_graph_list, test_graph_list
def separate_data(graph_list, fold_idx):
    train_idxes = np.loadtxt('/home/ln/spyder-workspace/HAP-master-for-graph-classification/data/%s/10fold_idx/train_idx-%d.txt' % (FLAGS.dataset, fold_idx), dtype=np.int32).tolist()
    test_idxes = np.loadtxt('/home/ln/spyder-workspace/HAP-master-for-graph-classification/data/%s/10fold_idx/test_idx-%d.txt' % (FLAGS.dataset, fold_idx), dtype=np.int32).tolist()
    return [graph_list[i] for i in train_idxes], [graph_list[i] for i in test_idxes]



