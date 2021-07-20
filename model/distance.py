# -*- coding: utf-8 -*-

from utils_model import get_root_path, exec, get_ts, save_npy
from nx_to_gxl import nx_to_gxl
from os.path import isfile
from os import getpid
import fileinput
import networkx as nx
import numpy as np



def ged(g1, g2, algo, debug=False, timeit=False):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()# ../src/graph-matching-toolkit
    append_str = get_append_str(g1, g2)#time stamp and the gid of two graphs
    src, t_datapath = setup_temp_data_folder(gp, append_str)#src is ../src/gmt_files;t_datapath is gp/data/temp_
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        raise RuntimeError(
            'Different meta data {} vs {}'.format(meta1, meta2))
    prop_file = setup_property_file(src, gp, meta1, append_str)
    rtn = []
    if not exec(
            'cd {} && java {}'
            ' -classpath {}/src/graph-matching-toolkit/bin algorithms.GraphMatching '
            './properties/properties_temp_{}.prop'.format(
                gp, '-XX:-UseGCOverheadLimit -XX:+UseConcMarkSweepGC -Xmx100g'
                if algo == 'astar' else '', get_root_path(), append_str)):
        rtn.append(-1)
    else:
        d, t, lcnt, g1size, g2size, result_file = get_result(gp, algo, append_str)
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up(t_datapath, prop_file, result_file)
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)

def normalized_ged(d, g1, g2):
    return 2 * d / (g1.number_of_nodes() + g2.number_of_nodes())


def normalized_ged3(d, g1, g2, g3):
    return 3 * d / (g1.number_of_nodes() + g2.number_of_nodes() + g3.number_of_nodes())


def unnormalized_ged(d, g1, g2):
    return d * (g1.number_of_nodes() + g2.number_of_nodes()) / 2


def setup_temp_data_folder(gp, append_str):
    tp = gp + '/data/temp_{}'.format(append_str)
    exec('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, append_str))
    return src, tp


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(g, g_name,
                                        '{}/{}.gxl'.format(tp, g_name))
    print('node_attres is {},edge_attrs is {}'.format(node_attres,edge_attrs))
    return algo + '_' + '_'.join(sorted(list(node_attres.keys())) + \
                                 sorted(list(edge_attrs.keys())))


def setup_property_file(src, gp, meta, append_str):
    destfile = '{}/properties/properties_temp_{}.prop'.format( \
        gp, append_str)
    srcfile = '{}/{}.prop'.format(src, meta)
    if not isfile(srcfile):
        if 'beam' in meta:  # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec('cp {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=':  # for beam
            print('s={}'.format(s))
        else:
            print(line.replace('temp', 'temp_{}'.format(append_str)))
    return destfile


def get_result(gp, algo, append_str):
    result_file = '{}/result/temp_{}'.format(gp, append_str)
    with open(result_file) as f:
        lines = f.readlines()
        ln = 16 if 'beam' in algo else 15
        t = int(lines[ln].split(': ')[1])  # msec
        ln = 23 if 'beam' in algo else 22
        d = float(lines[ln]) * 2  # alpha=0.5 --> / 2
        assert (d - int(d) == 0)
        d = int(d)
        if d < 0:
            d = -1  # in case rtn == -2
        ln = 26 if 'beam' in algo else 25
        g1size = int(lines[ln])
        ln = 27 if 'beam' in algo else 26
        g2size = int(lines[ln])
        ln = 28 if 'beam' in algo else 27
        lcnt = int(float(lines[ln]))
        return d, t, lcnt, g1size, g2size, result_file


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


def get_append_str(g1, g2):
    return '{}_{}_{}_{}'.format( \
        get_ts(), getpid(), g1.graph['gid'], g2.graph['gid'])


def clean_up(t_datapath, prop_file, result_file):
    for path in [t_datapath, prop_file, result_file]:
        exec('rm -rf {}'.format(path))


if __name__ == '__main__':
    from utils_model import load_data

    test_graphs = load_data('imdbmulti', train=False).graphs
    train_data = load_data('imdbmulti', train=True).graphs

    split = int(len(train_data) * (1 - 0.25))
    train_graphs = train_data[0:split]

    m = len(train_graphs)
    print('m:', m)
    n = len(train_graphs)
    np.set_printoptions(suppress=True, threshold=np.nan)
    ged_mat = np.zeros((m, n))
    time_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            g1 = train_graphs[i]
            g2 = train_graphs[j]
            ged_mat[i][j] = ged(g1, g2, 'beam80')

    print(ged_mat)
    save_npy('/home/ln/spyder-workspace/HAP-master/save/imdbmulti_calculate/imdbmulti_train(60)_train(60)_beam80', ged_mat)
    # save_npy('/home/ln/spyder-workspace/HAP-master/result/imdbmulti/time/ged_time_mat_imdbmulti_beam1', time_mat)

    # g1 = train_data.graphs[0]
    # g2 = test_data.graphs[0]
    # print(ged(g1, g2, 'beam5'))

    # g1 = test_data.graphs[15]
    # g2 = train_data.graphs[761]
    #
    # # nx.write_gexf(g1, get_root_path() + '/temp/g1.gexf')
    # # nx.write_gexf(g2, get_root_path() + '/temp/g2.gexf')
    # g1 = nx.read_gexf(get_root_path() + '/temp/g1_small.gexf')
    # g2 = nx.read_gexf(get_root_path() + '/temp/g2_small.gexf')
    # print(astar_ged(g1, g2))
    # print(beam_ged(g1, g2, 2))
