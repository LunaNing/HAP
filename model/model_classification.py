# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from classification import get_classification_labels_from_dist_mat, classify
from config import FLAGS
from dist_calculator import get_gs_dist_mat
from model import Model
from samplers import SelfShuffleList
from data_model import separate_data


class ClassificationModel(Model):
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        print('original_input_dim', self.input_dim)
        self.num_class = num_classes
        self.laplacians, self.features, self.adj_matrix, self.val_test_laplacians, self.val_test_features, self.val_test_adj_matrix, self.dropout = \
            self._create_basic_placeholders(FLAGS.batch_size)
        self.train_y_true_labels = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, self.num_class))
        self.val_test_y_true_labels = tf.placeholder(
            tf.float32, shape=(1, self.num_class))
        # Build the model.
        super(ClassificationModel, self).__init__()
        # self.cur_sample_class = 1  # 1 for pos, -1 for neg

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        # Transform the prediction score into classification score.
        assert (0 <= score <= 1)
        return score

    def get_feed_dict_for_train(self, train_graphs):
        rtn = {}
        y_true = np.zeros((FLAGS.batch_size, self.num_class))
        selected_idx = np.random.permutation(len(train_graphs))[:FLAGS.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        for i in range(FLAGS.batch_size):
            true_label = batch_graph[i].label
            y_true[i] = self._class_to_one_hot_label(true_label)
            # y_true[i] = tf.one_hot(indices=true_label, depth=self.num_class, on_value=1, off_value=0, axis=0)
        rtn[self.train_y_true_labels] = y_true
        rtn[self.dropout] = FLAGS.dropout
        return self._supply_etc_to_feed_dict(rtn, batch_graph, 'train')

    def get_feed_dict_for_val_test(self, graph, true_label):
        rtn = {}
        batch_graph = [graph]
        y_true = np.zeros((1, self.num_class))
        y_true[0] = self._class_to_one_hot_label(true_label)
        rtn[self.val_test_y_true_labels] = y_true
        return self._supply_etc_to_feed_dict(rtn, batch_graph, 'val_test')

    def get_true_sim(self, i, j, k, true_result):
        assert (true_result.dist_or_sim() == 'dist')
        _, d = true_result.dist_sim(i, j, k, FLAGS.dist_norm)
        c = classify(d, FLAGS.thresh_pos, FLAGS.thresh_neg)
        if c != 0:
            return c
        else:
            return None

    def get_eval_metrics_for_val(self):
        return ['loss', 'acc']

    def get_eval_metrics_for_test(self):
        return ['acc', 'pos_acc', 'neg_acc', 'prec@k', 'mrr', 'auc',
                'time', 'emb_vis_binary', 'emb_vis_gradual', 'ranking',
                'attention', 'draw_heat_hist', 'draw_gt_rk']

    def _get_determining_result_for_val(self):
        return 'val_acc'

    def _val_need_max(self):
        return True

    def _get_ins(self, layer, tvt):
        assert (layer.__class__.__name__ == 'GraphAttention' or
                layer.__class__.__name__ == 'Coarsening' or
                layer.__class__.__name__ == 'GraphConvolution')
        ins = []
        for features in (self._get_plhdr('features', tvt)):
            ins.append(features)
        # ins = self._get_plhdr('features', tvt)
        return ins

    def _val_test_pred_score(self):
        # assert (self.val_test_output.get_shape().as_list()[0] == 1)

        pred_score = tf.argmax(tf.nn.softmax(self.val_test_output), 1)
        # else:
        #     pred_score = tf.argmax(tf.nn.softmax(self.val_test_output), 1)

        return tf.squeeze(pred_score)

    def _task_loss(self, tvt):
        if tvt == 'train':
            y_pred = self._stack_concat(self.train_outputs)
            y_true = self.train_y_true_labels
        else:
            y_pred = self._stack_concat(self.val_test_output)
            y_true = self.val_test_y_true_labels
        print('y_true:', y_true.get_shape().as_list())
        print('y_pred:', y_pred.get_shape().as_list())
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_true, logits=y_pred)), \
               'cross_entropy_loss'


    @staticmethod
    def _load_pos_neg_train_triples(data, dist_calculator):
        gs = [g.nxgraph for g in data.train_graphs]
        dist_mat = get_gs_dist_mat(
            gs, gs, dist_calculator, 'train', 'train',
            FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo, FLAGS.dist_norm)
        _, _, _, pos_triples, neg_triples = \
            get_classification_labels_from_dist_mat(
                dist_mat, FLAGS.thresh_pos, FLAGS.thresh_neg)
        return SelfShuffleList(pos_triples), SelfShuffleList(neg_triples)

    def _sample_train_triple(self, data):
        if self.cur_sample_class == 1:
            li = self.pos_triples
            label = self._class_to_one_hot_label(self.cur_sample_class)
            self.cur_sample_class = -1
        elif self.cur_sample_class == -1:
            li = self.neg_triples
            label = self._class_to_one_hot_label(self.cur_sample_class)
            self.cur_sample_class = 1
        else:
            assert (False)
        x, y ,z = li.get_next()
        # print(x, y, z, label)
        return data.train_graphs[x], data.train_graphs[y], data.train_graphs[z], label

    def _class_to_one_hot_label(self, c):
        one_hot_label = np.zeros(self.num_class)
        one_hot_label[c] = 1
        return one_hot_label

    def _create_basic_placeholders(self, batch_size):
        laplacians = [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(1)] for _ in range(batch_size)]
        features = [tf.placeholder(tf.float32) for _ in range(batch_size)]
        adj_matrix = [tf.placeholder(tf.float32) for _ in range(batch_size)]
        val_test_laplacians = [[[tf.sparse_placeholder(tf.float32)] for _ in range(1)]]
        val_test_features = [tf.placeholder(tf.float32)]
        val_test_adj_matrix = [tf.placeholder(tf.float32)]
        dropout = tf.placeholder_with_default(0., shape=())
        return laplacians, features, adj_matrix, val_test_laplacians, val_test_features, val_test_adj_matrix, dropout

    def _supply_etc_to_feed_dict(self, feed_dict, batch_graph, tvt):
        for i, graph in enumerate(batch_graph):
            feed_dict[self._get_plhdr('features', tvt)[i]] = \
                graph.node_features
            feed_dict[self._get_plhdr('adj_matrix', tvt)[i]] = \
                graph.adj_matrix
            num_laplacians = 1
            for j in range(1):
                for k in range(num_laplacians):
                    feed_dict[self._get_plhdr('laplacians', tvt)[i][j][k]] = \
                        graph.get_laplacians(j)[k]
        return feed_dict
