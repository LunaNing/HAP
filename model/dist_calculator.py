# -*- coding: utf-8 -*-

from utils_model import get_save_path, get_result_path, \
    save, load, load_data, get_norm_str, create_dir_if_not_exists, save_npy
from distance import ged, normalized_ged
from collections import OrderedDict
import numpy as np


class DistCalculator(object):
    def __init__(self, dataset, dist_metric, algo):
        self.sfn = '{}/{}_{}_{}{}_gidpair_dist_map'.format(
            get_save_path(), dataset, dist_metric, algo,
            '' if algo == 'astar' else '_revtakemin')
        self.algo = algo
        self.gidpair_dist_map = load(self.sfn)
        if not self.gidpair_dist_map:
            self.gidpair_dist_map = OrderedDict()
            save(self.sfn, self.gidpair_dist_map)
            print('Saved dist map to {} with {} entries'.format(
                self.sfn, len(self.gidpair_dist_map)))
        else:
            print('Loaded dist map from {} with {} entries'.format(
                self.sfn, len(self.gidpair_dist_map)))
        if dist_metric == 'ged':
            self.dist_func = ged
        else:
            raise RuntimeError('Unknwon distance metric {}'.format(dist_metric))

    def calculate_dist(self, g1, g2):
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        d = self.gidpair_dist_map.get(pair)
        if d is None:
            rev_pair = (gid2, gid1)
            rev_d = self.gidpair_dist_map.get(rev_pair)
            if rev_d:
                d = rev_d
            else:
                d = self.dist_func(g1, g2, self.algo)
                if self.algo != 'astar':
                    d = min(d, self.dist_func(g2, g1, self.algo))
            self.gidpair_dist_map[pair] = d
            print('{}Adding entry ({}, {}) to dist map'.format(
                ' ' * 80, pair, d))
            save(self.sfn, self.gidpair_dist_map)
        return d, normalized_ged(d, g1, g2)

    def load_from_dist_mat(self, mats, row_gs, col_gs, check_symmetry=False):
        """
        Load the internal distance map from an external distance matrix,
            which is assumed to be m by n,
        Use this function if the pairwise distances have been calculate
            elsewhere, e.g. by the multiprocessing version of
            running the baselines as in one of the functions in exp.py.
        The distance map stored in this distance calculator will be
            enriched/expanded by the results.
        :param mat: a list of m by n distance matrix.
        :param row_gs: the corresponding row graphs
        :param col_gs: the corresponding column graphs
        :param check_symmetry: whether to check if mat if symmetric or not
        :return:
        """
        assert (mats)
        m, n = mats[0].shape
        assert (m == len(row_gs) and n == len(col_gs))
        for i in range(m):
            for j in range(n):
                d = np.min([mat[i][j] for mat in mats])
                if check_symmetry:
                    d_t = np.min(mat[j][i] for mat in mats)
                    if d != d_t:
                        raise RuntimeError(
                            'Asymmetric distance {} {}: {} and {}'.format(
                                i, j, d, d_t))
                gid1 = row_gs[i].graph['gid']
                gid2 = col_gs[j].graph['gid']
                pair = (gid1, gid2)
                d_m = self.gidpair_dist_map.get(pair)
                if d_m:
                    if d != d_m:
                        raise RuntimeError(
                            'Inconsistent distance {} {}: {} and {}'.format(
                                i, j, d, d_m))
                else:
                    self.gidpair_dist_map[pair] = d
        save(self.sfn, self.gidpair_dist_map)


def get_train_train_dist_mat(dataset, dist_metric, dist_algo, norm):
    train_data = load_data(dataset, train=True)
    gs = train_data.graphs
    dist_calculator = DistCalculator(dataset, dist_metric, dist_algo)
    return get_gs_dist_mat(gs, gs, dist_calculator, 'train', 'train', dataset,
                           dist_metric, dist_algo, norm)


def get_gs_dist_mat(gs1, gs2, dist_calculator, tvt1, tvt2,
                    dataset, dist_metric, dist_algo, norm):
    mat_str = '{}({})_{}({})'.format(tvt1, len(gs1), tvt2, len(gs2))
    dir = '{}/dist_mat'.format(get_save_path())
    create_dir_if_not_exists(dir)
    sfn = '{}/{}_{}_dist_mat_{}{}_{}'.format(
        dir, dataset, mat_str, dist_metric,
        get_norm_str(norm), dist_algo)
    l = load(sfn)
    if l is not None:
        print('Loaded from {}'.format(sfn))
        return l
    m = len(gs1)
    n = len(gs2)
    dist_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            g1 = gs1[i]
            g2 = gs2[j]
            d, normed_d = dist_calculator.calculate_dist(g1, g2)
            if norm:
                dist_mat[i][j] = normed_d
            else:
                dist_mat[i][j] = d
    save(sfn, dist_mat)
    print('Saved to {}'.format(sfn))
    return dist_mat


def get_gs_triple_dist_mat(gs1, gs2, dist_calculator, tvt1, tvt2,
                    dataset, dist_metric, dist_algo, norm):
    mat_str = '{}({})_{}({})'.format(tvt1, len(gs1), tvt2, len(gs2))
    dir = '{}/dist_mat'.format(get_save_path())
    create_dir_if_not_exists(dir)
    sfn = '{}/{}_{}_dist_mat_{}{}_{}'.format(
        dir, dataset, mat_str, dist_metric,
        get_norm_str(norm), dist_algo)
    l = load(sfn)
    if l is not None:
        print('Loaded from {}'.format(sfn))
        return l
    m = len(gs1)
    n = len(gs2)
    dist_mat = np.zeros((m, n, n))
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if k != j:
                    g1 = gs1[i]
                    g2 = gs2[j]
                    g3 = gs2[k]
                    d1, normed_d1 = dist_calculator.calculate_dist(g1, g2)
                    d2, normed_d2 = dist_calculator.calculate_dist(g1, g3)
                    if norm:
                        dist_mat[i][j][k] = normed_d1 - normed_d2
                    else:
                        dist_mat[i][j][k] = d1 -d2
    save(sfn, dist_mat)
    print('Saved to {}'.format(sfn))
    return dist_mat


if __name__ == '__main__':
    dataset = 'imdbmulti'
    dist_metric = 'ged'
    dist_algo = 'beam80'
    dist_calculator = DistCalculator(dataset, dist_metric, dist_algo)
    # get_train_train_dist_mat(dataset, dist_metric, dist_algo, norm=True)

    # The server qilin calculated all the pairwise distances between
    # the training graphs.
    # Thus, enrich the distance map (i.e. calculator) using the qilin results.
    mat1 = np.load('/home/ln/spyder-workspace/HAP-master/save/dsit_mat/imdbmulti_train(60)_train(60)_dist_mat_ged_norm_beam40.')
    mat2 = np.load('/home/ln/spyder-workspace/HAP-master/save/dsit_mat/imdbmulti_train(60)_train(60)_hungarian.npy')
    mat3 = np.load('/home/ln/spyder-workspace/HAP-master/save/dsit_mat/imdbmulti_train(60)_train(60)_vj.npy')
    row_gs = load_data(dataset, train=True).graphs[0:60]
    col_gs = load_data(dataset, train=True).graphs[0:60]
    dist_calculator.load_from_dist_mat([mat1, mat2, mat3], row_gs, col_gs,
                                       check_symmetry=False)
    # m, n = mat1.shape
    # mat = np.zeros((m,n))
    # assert (m == len(row_gs) and n == len(col_gs))
    # for i in range(m):
    #     for j in range(n):
    #         d = np.min(mat1[i][j]+mat2[i][j]+mat3[i][j])
    #         mat[i][j] = d
    # save('/home/ln/spyder-workspace/HAP-master/save/dist_mat/imdbmulti_train(60)_train(60)_dist_mat_ged_norm_beam80.pickle', mat)