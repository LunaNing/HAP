# -*- coding: utf-8 -*-
from utils_model import load, save,load_joblib, save_joblib, get_save_path, get_data_path, get_train_str, sorted_nicely
from glob import glob
from os.path import basename
import networkx as nx
import binascii
import numpy as np


class Data(object):
    def __init__(self, name_str):
        name = self.__class__.__name__ + '_' + name_str + self.name_suffix()
        self.name = name
        sfn = self.save_filename(self.name)
        print('file name:', sfn)
        temp = load(sfn)
        if temp:
            self.__dict__ = temp
            print('{} loaded from {}{}'.format(name, sfn, 'with {} graphs'.format(len(self.graphs)) if hasattr(self,
                                                                                                               'graphs') else ''))
        else:
            self.init()
            save(sfn, self.__dict__)
            print('{} saved to {}'.format(name, sfn))

    def init(self):
        raise NotImplementedError()

    def name_suffix(self):
        return ''

    def save_filename(self, name):
        return '{}/{}'.format(get_save_path(), name)

    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]


class AIDSData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/{}/{}'.format(get_data_path(), self.get_folder_name(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))
        if 'nef' in self.get_folder_name():
            print('Romoving edge features')
            for g in self.graphs:
                self.remove_valence(g)

    def get_folder_name(self):
        raise NotImplementedError()

    def remover_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS700nefData(AIDSData):
    def get_folder_name(self):
        return 'AIDS700nef'

    def remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class IMDBMultiData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/IMDBMulti/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))


class LinuxData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/LINUX/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))

class MiviaData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/MIVIA/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs_from_binary(datadir + '/*')
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))

def iterate_get_graphs(dir):
    graphs = []
    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs

def iterate_get_graphs_from_binary(dir):
    graphs = []
    for file in sorted_nicely(glob(dir)):
        adj_matrix = read_unlabelled_graph(file)
        g = nx.from_numpy_matrix(adj_matrix)
        graphs.append(g)
        gid = basename(file).split('.')[1]
        g.graph['gid'] = gid
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs

def read_unlabelled_graph(filename):
    # print("Reading graph: " + filename)
    with open(filename, "rb") as f:
        length = get_next_word(f)
        # print('length:', length)

        # mx = [[0 for _ in range(length)] for _ in range(length)]
        mx = np.zeros([length, length])

        # print(mx)
        for node in range(length):
            for _ in range(get_next_word(f)):
                mx[node][get_next_word(f)] = 1
                # print(mx)

    return mx


def get_next_word(f):
    lower = f.read(1)
    higher = f.read(1)
    strHexLower = str(binascii.hexlify(lower),'utf-8')
    strHexHigher = str(binascii.hexlify(higher),'utf-8')
    return int(strHexHigher + strHexLower, 16)

if __name__ == '__main__':
    from utils_model import load_data
    nn = []
    data = load_data('imdbmulti', True)
    for g in data.graphs:
        nn.append(g.number_of_nodes())
        print(g.graph['gid'], g.number_of_nodes())
    print(max(nn))