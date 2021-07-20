from warnings import warn
import numpy as np
import tensorflow as tf

from config import FLAGS
from layer_factory import create_layers


def get_ini_adj(self, tvt):
    adj = []
    for adj_matrix in (self._get_plhdr('adj_matrix', tvt)):
        adj.append(adj_matrix)
    return adj


def get_num_nonzero(self, tvt):
    num_nonzero = self._get_plhdr('num_nonzero_1', tvt) + self._get_plhdr('num_nonzero_2', tvt) + self._get_plhdr(
        'num_nonzero_3',
        tvt)
    return num_nonzero


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.layers = []
        self.train_loss = 0
        self.val_test_loss = 0
        self.optimizer = None
        self.opt_op = None

        self.batch_size = FLAGS.batch_size
        self.weight_decay = FLAGS.weight_decay
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self._build()
        print('Flow built')
        # Build metrics
        self._loss()
        print('Loss built')
        self.opt_op = self.optimizer.minimize(self.train_loss)
        print('Optimizer built')

    def _build(self):
        # Create layers according to FLAGS.
        self.layers = create_layers(self)
        assert (len(self.layers) > 0)
        print('Created {} layers: {}'.format(
            len(self.layers), ', '.join(l.get_name() for l in self.layers)))

        # Build the siamese model for train and val_test, respectively,
        coarsen_count = 0
        for tvt in ['train', 'val_test']:
            print(tvt)
            orig_adj = get_ini_adj(self, tvt)
            adj = get_ini_adj(self, tvt)
            # Go through each layer except the last one.
            acts = [self._get_ins(self.layers[0], tvt)]
            # print('acts[0][0]:', acts[0][0])
            outs = None
            graph_emb = None
            graph_emb1 = []
            graph_emb2 = []
            graph_emb3 = None
            adj1 = None
            att_mat = None
            coarsen_graphs=[]
            for k, layer in enumerate(self.layers):
                print(layer.get_name())
                ins = self._proc_ins(acts[-1], adj, k, layer, tvt)
                # print('ins[0]:', ins[0])
                if layer.__class__.__name__ == 'GraphAttention':
                    outs = layer(ins)
                elif layer.__class__.__name__ == 'Coarsening':
                    feature, adj_matrix, att= layer(ins)
                    outs = feature
                    adj = adj_matrix
                    # graph_embs=[]
                    # for i in range(len(outs)):
                    #     graph_emb = tf.reduce_sum(outs[i],axis=0)
                    #     graph_embs.append(graph_emb)
                    # coarsen_graphs.append(graph_embs)
                    if coarsen_count == 0:
                        graph_emb1 = tf.reduce_sum(outs,axis=0)
                        adj1 = adj
                        att_mat = att
                    if coarsen_count == 1:
                        graph_emb2 = tf.reduce_sum(outs,axis=0)
                    # if coarsen_count == 2:
                    #     graph_emb3 = outs
                    coarsen_count += 1
                    # graph_emb = outs
                elif layer.__class__.__name__ == 'Average':
                    outs = layer(ins)
                    graph_emb = outs
                elif layer.__class__.__name__ == 'Attention':
                    outs = layer(ins)
                    graph_emb = outs
                elif layer.__class__.__name__ == 'GraphConvolution':
                    outs = layer(ins)
                elif layer.__class__.__name__ == 'GraphConvolutionTopk':
                    outs = layer(ins)
                    graph_emb = outs
                elif layer.__class__.__name__ == 'GraphConvolutionCoarsen':
                    outs = layer(ins)
                    graph_emb = outs
                elif layer.__class__.__name__ == 'NTN':
                    outs = layer(ins)
                elif layer.__class__.__name__ == 'Dense':
                    outs = layer(ins)
                    print('outs:', outs)

                # outs = self._proc_outs(outs, k, layer, tvt)
                acts.append(outs)
            #thresh = tf.zeros([1])
            if tvt == 'train':
                self.train_outputs = outs
                self.train_acts = acts
            else:
                # self.graph_embedding1 = tf.reshape(tf.reduce_mean(graph_emb1, 0), [1, -1])
                # self.graph_embedding2 = tf.reshape(tf.reduce_mean(graph_emb2, 0), [1, -1])

                self.graph_embedding = graph_emb
                self.orig_adj = orig_adj
                self.targ_adj = adj1
                self.att_mat = att_mat
                self.val_test_output = outs
                self.val_test_pred_score = self._val_test_pred_score()
                self.val_test_acts = acts
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        self.train_loss = self._loss_helper('train')
        self.val_test_loss = self._loss_helper('val')

    def get_act(self):
        raise NotImplementedError()

    def pred_sim_without_act(self):
        raise NotImplementedError()

    def apply_final_act_np(self, score):
        raise NotImplementedError()

    def get_feed_dict_for_train(self, data):
        raise NotImplementedError()

    def get_feed_dict_for_val_test_regression(self, g1, g2, g3, true_sim):
        raise NotImplementedError()

    def get_feed_dict_for_val_test_classification(self, g1, g2, true_sim):
        raise NotImplementedError()

    def get_true_sim(self, i, j, true_result):
        raise NotImplementedError()

    def get_eval_metrics_for_val(self):
        raise NotImplementedError()

    def get_eval_metrics_for_test(self):
        raise NotImplementedError()

    def _get_determining_result_for_val(self):
        raise NotImplementedError()

    def _val_need_max(self):
        raise NotImplementedError()

    def find_load_best_model(self, sess, saver, val_results_dict):
        cur_max_metric = -float('inf')
        cur_min_metric = float('inf')
        cur_best_iter = 1
        metric_list = []
        early_thresh = int(FLAGS.iters * 0.1)
        deter_r_name = self._get_determining_result_for_val()
        for iter, val_results in val_results_dict.items():
            metric = val_results[deter_r_name]
            metric_list.append(metric)
            if iter >= early_thresh:
                if self._val_need_max():
                    if metric >= cur_max_metric:
                        cur_max_metric = metric
                        cur_best_iter = iter
                else:
                    if metric <= cur_min_metric:
                        cur_min_metric = metric
                        cur_best_iter = iter
        if self._val_need_max():
            argfunc = np.argmax
            takefunc = np.max
            best_metric = cur_max_metric
        else:
            argfunc = np.argmin
            takefunc = np.min
            best_metric = cur_min_metric
        global_best_iter = list(val_results_dict.items()) \
            [int(argfunc(metric_list))][0]
        global_best_metirc = takefunc(metric_list)
        if global_best_iter != cur_best_iter:
            warn(
                'The global best iter is {} with {}={:.5f},\nbut the '
                'best iter after first 10% iterations is {} with {}={:.5f}'.format(
                    global_best_iter, deter_r_name, global_best_metirc,
                    cur_best_iter, deter_r_name, best_metric))
        lp = '{}/models/{}.ckpt'.format(saver.get_log_dir(), cur_best_iter)
        self.load(sess, lp)
        print('Loaded the best model at iter {} with {} {:.5f}'.format(
            cur_best_iter, deter_r_name, best_metric))
        return cur_best_iter
        # return None

    def _get_ins(self, layer, tvt):
        raise NotImplementedError()

    def _supply_etc_to_ins(self, ins, adj):
        for i, item in \
                enumerate(adj):
            ins[i] = [ins[i], item]
        # ins = [ins, adj]
        return ins

    def _supply_laplacians_etc_to_ins(self, ins, adj, tvt, gcn_id):
        for i, laplacians in \
                enumerate(self._get_plhdr('laplacians', tvt)):
            ins[i] = [ins[i], laplacians[0]]
        return ins

    def _supply_del_to_ins(self, ins):
        for i, inss in enumerate(ins):
            ins[i] = [inss[0]]
        return ins

    def _proc_ins_for_merging_layer(self, ins, tvt):
        raise NotImplementedError()

    def _val_test_pred_score(self):
        raise NotImplementedError()

    def _task_loss(self, tvt):
        raise NotImplementedError()

    def _proc_ins(self, ins, adj, k, layer, tvt):
        ln = layer.__class__.__name__
        ins_mat = None
        if k != 0 and tvt == 'train':
            # sparse matrices (k == 0; the first layer) cannot be logged.
            need_log = True
        else:
            need_log = False

        if ln == 'GraphConvolution' or ln == 'GraphConvolutionTopk' or ln == 'GraphConvolutionCoarsen':
            gcn_count = int(layer.name.split('_')[-1])
            assert (gcn_count >= 1)  # 1-based
            gcn_id = gcn_count - 1
            ins = self._supply_laplacians_etc_to_ins(ins, adj, tvt, gcn_id)
        elif ln == 'Dense':
            ins = self._stack_concat(ins)
        else:
            ins = self._supply_etc_to_ins(ins, adj)

        return ins

    def _proc_outs(self, outs, k, layer, tvt):
        outs_mat = self._stack_concat(outs)
        ln = layer.__class__.__name__
        if tvt == 'train':
            self._log_mat(outs_mat, layer, 'outs')
        if tvt == 'val_test' and layer.produce_graph_level_emb():
            if ln != 'ANPM' and ln != 'ANPMD' and ln != 'ANNH':
                embs = outs
            else:
                embs = layer.embeddings
            assert (len(embs) == 3)
            # Note: some architecture may NOT produce
            # any graph-level embeddings.
            self.graph_embeddings = embs
            s = embs[0].get_shape().as_list()
            assert (s[0] == 1)
            self.embed_dim = s[1]

        return outs

    def _get_plhdr(self, key, tvt):
        if tvt == 'train':
            return self.__dict__[key]
        else:
            assert (tvt == 'test' or tvt == 'val' or tvt == 'val_test')
            return self.__dict__['val_test_' + key]

    def _get_last_coarsen_layer_outputs(self, tvt):
        acts = self.train_acts if tvt == 'train' else self.val_test_acts
        assert (len(acts) == len(self.layers) + 1)
        idx = None
        for k, layer in enumerate(self.layers):
            if 'Graph' not in layer.__class__.__name__:
                idx = k
                break
        assert (idx)
        return acts[idx]

    def _stack_concat(self, x):
        if type(x) is list:
            list_of_tensors = x
            assert (list_of_tensors)

            s = list_of_tensors[0].get_shape()
            if s != ():
                return tf.concat(list_of_tensors, 0)
            else:
                return tf.stack(list_of_tensors)
        else:
            # assert(len(x.get_shape()) == 2) # should be a 2-D matrix
            return x

    def _log_mat(self, mat, layer, label):
        tf.summary.histogram(layer.name + '/' + label, mat)

    def save(self, sess, saver, iter):
        logdir = saver.get_log_dir()
        sp = '{}/models/{}.ckpt'.format(logdir, iter)
        tf.train.Saver(self.vars).save(sess, sp)

    def load(self, sess, load_path):
        tf.train.Saver(self.vars).restore(sess, load_path)

    def _loss_helper(self, tvt):
        rtn = 0

        # weight decay loss
        wdl = 0
        for layer in self.layers:
            for var in layer.vars.values():
                wdl = self.weight_decay * tf.nn.l2_loss(var)
                rtn += wdl
        if tvt == 'train':
            tf.summary.scalar('weight_decay_loss', wdl)

        loss, loss_label = self._task_loss(tvt)
        rtn += loss
        if tvt == 'train':
            tf.summary.scalar(loss_label, loss)
        return rtn
