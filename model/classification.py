import numpy as np


def get_classification_labels_from_dist_mat(dist_mat, thresh_pos, thresh_neg):
    print('dist_mat.shape:', dist_mat.shape)
    m, n, l = dist_mat.shape
    label_mat = np.zeros((m, n, l))
    num_poses = 0
    num_negs = 0
    pos_triples = []
    neg_triples = []
    for i in range(m):
        num_pos = 0
        num_neg = 0
        for j in range(n):
            for k in range(n):
                d = dist_mat[i][j][k]
                c = classify(d, thresh_pos, thresh_neg)
                if c == 1:
                    label_mat[i][j][k] = 1
                    num_pos += 1
                    pos_triples.append((i, j, k))
                elif c == -1:
                    label_mat[i][j][k] = -1
                    num_neg += 1
                    neg_triples.append((i, j, k))
        num_poses += num_pos
        num_negs += num_neg
    return label_mat, num_poses, num_negs, pos_triples, neg_triples


def classify(dist, thresh_pos, thresh_neg):
    if dist > thresh_pos:
        return 1
    elif dist < thresh_neg:
        return -1
    else:
        return 0
