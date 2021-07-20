# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from config import FLAGS
from inits import xavier, zeros, glorot
from utils_model import dot

_LAYER_UIDS = {}  # dictionary
_LAYERS = []  # list
conv1d = tf.layers.conv1d


def get_layer_name(layer):
    layer_name = layer.__class__.__name__.lower()
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        layer_id = 1
    else:
        _LAYER_UIDS[layer_name] += 1
        layer_id = _LAYER_UIDS[layer_name]
    _LAYERS.append(layer)
    return str(len(_LAYERS)) + '_' + layer_name + '_' + str(layer_id)


def sparse_dropout(x, keep_prob, noise_shape):
    """ Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class Layer(object):
    """define the basic layer class, layers(coarsening layer and graph attention layer) are implemented follow the layer class"""

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument:' + kwarg
        name = kwargs.get('name')
        if not name:
            name = get_layer_name(self)
        self.name = name
        self.vars = {}
        self.sparse_inputs = False

    def get_name(self):
        return self.name

    @staticmethod
    def produce_graph_level_emb():
        return False

    def __call__(self, inputs):
        return self._call(inputs)

    def _call(self, inputs):
        raise NotImplementedError

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '_weights/' + var,
                                 self.vars[var])  # the first parameter means name,the second is the value of that name

    def handle_dropout(self, dropout_bool):
        if dropout_bool:
            self.dropout = FLAGS.dropout
        else:
            self.dropout = 0


class Dense(Layer):
    """ Dense layer. """

    def __init__(self, input_dim, output_dim, dropout, act, bias, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.bias = bias
        self.act = act

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self._log_vars()

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, dropout, sparse_inputs, act, bias, featureless, num_supports, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.support = None
        self.act = act

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_supports):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            rt = []
            for input in inputs:
                assert (len(input) == 2)
                rt.append(self._call_one_graph(input))
            return rt
        else:
            assert (len(inputs) == 2)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, input):
        x = input[0]
        # num_features_nonzero = input[2]
        self.laplacians = input[1]

        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        support_list = []
        for i in range(len(self.laplacians)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_]' + str(i)]
            support = dot(self.laplacians[i], pre_sup, sparse=True)
            support_list.append(support)
        output = tf.add_n(support_list)  # the elements of the list add one by one

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolutionTopk(Layer):
    def __init__(self, input_dim, coarsen_dim, dropout, sparse_inputs, act, bias, featureless, num_supports, **kwargs):
        super(GraphConvolutionTopk, self).__init__(**kwargs)
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.support = None
        self.act = act
        self.coarsen_dim = coarsen_dim

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_supports):
                self.vars['weights_' + str(i)] = glorot([input_dim, 1], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([1], name='bias')
        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            rt = []
            for input in inputs:
                assert (len(input) == 2)
                rt.append(self._call_one_graph(input))
            return rt
        else:
            assert (len(inputs) == 2)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, input):
        x = input[0]
        self.laplacians = input[1]

        x = tf.nn.dropout(x, 1 - self.dropout)

        support_list = []
        for i in range(len(self.laplacians)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_]' + str(i)]
            support = dot(self.laplacians[i], pre_sup, sparse=True)
            support_list.append(support)
        score = tf.add_n(support_list)  # the elements of the list add one by one

        if self.bias:
            score += self.vars['bias']

        score = self.act(score)
        values, idx = tf.nn.top_k(tf.transpose(score), int(self.coarsen_dim))
        values = tf.transpose(values)
        new_x= tf.squeeze(tf.gather(x, idx))
        # values = tf.expand_dims(values, -1)
        new_x = tf.multiply(new_x, values)
        return new_x

class GraphConvolutionAttention(GraphConvolution):
    """ Graph convolution with attention layer. """

    def __init__(self, input_dim, output_dim, dropout, sparse_inputs, act,
                 bias, featureless, num_supports, **kwargs):
        super(GraphConvolutionAttention, self).__init__(input_dim, output_dim,
                                                        dropout, sparse_inputs, act, bias, featureless, num_supports,
                                                        **kwargs)

    def _call_one_graph(self, input):
        x = super(GraphConvolutionAttention, self)._call_one_graph(input)
        L = tf.sparse_tensor_dense_matmul(self.laplacians[0], tf.eye(tf.shape(x)[0]))
        degree_att = -tf.log(tf.reshape(tf.diag_part(L), [-1, 1]))
        output = tf.multiply(x, degree_att)
        return output

class Merge(Layer):
    """ Merge layer. """

    def __init__(self, **kwargs):
        super(Merge, self).__init__(**kwargs)

    def produce_graph_level_emb(self):
        return False

    def merge_graph_level_embs(self):
        return True

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 2)
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            assert (len(inputs) == 2)
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        raise NotImplementedError()

class NTN(Merge):
    """ NTN layer.
    (Socher, Richard, et al.
    "Reasoning with neural tensor networks for knowledge base completion."
    NIPS. 2013.). """

    def __init__(self, input_dim, feature_map_dim, apply_u, dropout,
                 inneract, bias, **kwargs):
        super(NTN, self).__init__(**kwargs)

        self.feature_map_dim = feature_map_dim
        self.apply_u = apply_u
        self.bias = bias
        self.inneract = inneract

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['V'] = glorot([feature_map_dim, input_dim * 2], name='V')
            self.vars['W'] = glorot([feature_map_dim, input_dim, input_dim],
                                    name='W')
            if self.bias:
                self.vars['b'] = zeros([feature_map_dim], name='b')
            if self.apply_u:
                self.vars['U'] = glorot([feature_map_dim, 1], name='U')

        self._log_vars()

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]

        # dropout
        x_1 = tf.nn.dropout(x_1, 1 - self.dropout)
        x_2 = tf.nn.dropout(x_2, 1 - self.dropout)

        # one pair comparison
        return interact_two_sets_of_vectors(
            x_1, x_2, self.feature_map_dim,
            V=self.vars['V'],
            W=self.vars['W'],
            b=self.vars['b'] if self.bias else None,
            act=self.inneract,
            U=self.vars['U'] if self.apply_u else None)


def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    for i in range(interaction_dim):
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = tf.multiply(tf.ones_like(x_1), x_2)
            concat = tf.concat([x_1, tiled_x_2], 1)
            v_weight = tf.reshape(V[i], [-1, 1])
            V_out = tf.matmul(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = tf.matmul(x_1, W[i])
            h = tf.matmul(temp, tf.transpose(x_2))  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)

    output = tf.concat(feature_map, 1)
    if act is not None:
        output = act(output)
    if U is not None:
        output = tf.matmul(output, U)

    return output



class Average(Layer):
    """ Average layer. """

    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)

    def produce_graph_level_emb(self):
        return True

    def merge_graph_level_embs(self):
        return False

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, input):
        x = input[0]  # (N, D)
        output = tf.reshape(tf.reduce_sum(x, 0), [1, -1])  # (1, D)
        return output


class Attention(Average):
    """ Attention layer."""

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.emb_dim = input_dim  # same dimension D as input embeddings
        self.att_times = att_times
        self.att_num = att_num
        self.att_style = att_style
        self.att_weight = att_weight
        assert (self.att_times >= 1)
        assert (self.att_num >= 1)
        assert (self.att_style == 'dot' or self.att_style == 'slm' or
                'ntn_' in self.att_style)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.att_num):
                self.vars['W_' + str(i)] = \
                    glorot([self.emb_dim, self.emb_dim],
                           name='W_' + str(i))
                if self.att_style == 'slm':
                    self.interact_dim = 1
                    self.vars['NTN_V_' + str(i)] = \
                        glorot([self.interact_dim, 2 * self.emb_dim],
                               name='NTN_V_' + str(i))
                if 'ntn_' in self.att_style:
                    self.interact_dim = int(self.att_style[4])
                    self.vars['NTN_V_' + str(i)] = \
                        glorot([self.interact_dim, 2 * self.emb_dim],
                               name='NTN_V_' + str(i))
                    self.vars['NTN_W_' + str(i)] = \
                        glorot([self.interact_dim, self.emb_dim, self.emb_dim],
                               name='NTN_W_' + str(i))
                    self.vars['NTN_U_' + str(i)] = \
                        glorot([self.interact_dim, 1],
                               name='NTN_U_' + str(i))
                    self.vars['NTN_b_' + str(i)] = \
                        zeros([self.interact_dim],
                              name='NTN_b_' + str(i))

        self._log_vars()

    def produce_node_atts(self):
        return True

    def _call_one_mat(self, inputs):
        #outputs = []
        output = None
        x = inputs[0]
        for i in range(self.att_num):
            #acts = [inputs]
            assert (self.att_times >= 1)
            for _ in range(self.att_times):
                #x = acts[-1]  # x is N*D
                temp = tf.reshape(tf.reduce_mean(x, 0), [1, -1])  # (1, D)
                h_avg = tf.tanh(tf.matmul(temp, self.vars['W_' + str(i)])) if \
                    self.att_weight else temp
                self.att = self._gen_att(x, h_avg, i)
                output = tf.matmul(tf.reshape(self.att, [1, -1]), x)  # (1, D)
                x_new = tf.multiply(x, self.att)
                #acts.append(x_new)
            #outputs.append(output)
        return output

    def _gen_att(self, x, h_avg, i):
        if self.att_style == 'dot':
            return interact_two_sets_of_vectors(
                x, h_avg, 1,  # interact only once
                W=[tf.eye(self.emb_dim)],
                act=tf.sigmoid)
        elif self.att_style == 'slm':
            # return tf.sigmoid(tf.matmul(concat, self.vars['a_' + str(i)]))
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                act=tf.sigmoid)
        else:
            assert ('ntn_' in self.att_style)
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                W=self.vars['NTN_W_' + str(i)],
                b=self.vars['NTN_b_' + str(i)],
                act=tf.sigmoid,
                U=self.vars['NTN_U_' + str(i)])


""" ### End of generating node embeddings into graoh-level embeddings. ### """
def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    for i in range(interaction_dim):
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = tf.multiply(tf.ones_like(x_1), x_2)
            concat = tf.concat([x_1, tiled_x_2], 1)
            v_weight = tf.reshape(V[i], [-1, 1])
            V_out = tf.matmul(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = tf.matmul(x_1, W[i])
            h = tf.matmul(temp, tf.transpose(x_2))  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)

    output = tf.concat(feature_map, 1)
    if act is not None:
        output = act(output)
    if U is not None:
        output = tf.matmul(output, U)

    return output



class GraphAttention(Layer):
    def __init__(self, input_dim, output_dim, dropout, sparse_inputs, act, bias, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.bias = bias
        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['W'] = glorot([input_dim, output_dim], name='attention_W')
            self.vars['a'] = glorot([2*output_dim, 1], name='attention_a')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            rt = []
            for input in inputs:
                assert (len(input) == 2)
                rt.append(self._call_one_graph(input))
            return rt
        else:
            assert (len(inputs) == 2)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        # num_features_nonzero = inputs[2]

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        # else:
        if self.dropout != 0:
            x = tf.nn.dropout(x, 1 - self.dropout)

        z = dot(x, self.vars['W'], sparse=self.sparse_inputs)
        node_num = tf.shape(x)[0]

        # zizj = tf.concat([np.repeat(z, node_num, axis=1).reshape(node_num * node_num, -1), np.repeat(z, node_num, axis=0)], 1).reshape(node_num, -1, 2*self.output_dim)
        # zizj = tf.concat(
        #     [tf.tile(z, [1, node_num * node_num]).reshape(node_num * node_num, -1),
        #      tf.tile(z, [node_num * node_num, 1])],
        #     1).reshape(node_num, -1, 2 * self.output_dim)

        c1 = tf.reshape(tf.tile(z, [1, node_num]), [node_num*node_num,-1])
        c2 = tf.tile(z, [node_num, 1])
        concat= tf.concat([c1, c2], 1)
        #zizj = tf.reshape(tf.concat([c1, c2], 1), [node_num, -1, 2*self.output_dim])
        zizj = tf.reshape(concat, [-1, 2*self.output_dim])

        # e = tf.nn.leaky_relu(tf.reshape(tf.squeeze(dot(zizj, self.vars['a'], sparse=False)), [node_num, node_num]))
        e = tf.nn.leaky_relu(tf.reshape(
            tf.squeeze(dot(tf.reshape(zizj, [node_num * node_num, 2 * self.output_dim]), self.vars['a'], sparse=False),
                       axis=1), [node_num, node_num]))

        zero_vec = -9e15*tf.ones_like(e)
        alpha = tf.where(adj > 0, e, zero_vec)

        alpha = tf.nn.softmax(alpha, axis=1)
        if self.dropout != 0:
            alpha = tf.nn.dropout(alpha, 1 - self.dropout)

        output = dot(alpha, z, sparse=False)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Coarsening(Layer):
    def __init__(self, input_dim, coarsen_dim, dropout, act, bias, **kwargs):
        super(Coarsening, self).__init__(**kwargs)
        self.coarsen_dim = coarsen_dim
        self.act = act
        self.bias = bias
        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['W'] = glorot([input_dim, coarsen_dim], name='attention_W')
            self.vars['a'] = glorot([2*coarsen_dim, 1], name='attention_a')
            if self.bias:
                self.vars['bias'] = zeros([coarsen_dim], name='bias')
        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            feature = []
            adj_matrix = []
            att_matrix = []
            for input in inputs:
                assert (len(input) == 2)
                x, adj, att= self._call_one_graph(input)
                feature.append(x)
                adj_matrix.append(adj)
                att_matrix.append(att)
                # rt.append(self._call_one_graph(input))
            return feature, adj_matrix, att_matrix
        else:
            print('length of inputs:', len(inputs))
            assert (len(inputs) == 2)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        # num_features_nonzero = inputs[2]

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        # else:
        if self.dropout != 0:
            x = tf.nn.dropout(x, 1 - self.dropout)

        z = dot(x, self.vars['W'], sparse=self.sparse_inputs)
        node_num = tf.shape(x)[0]

        # zizj = tf.concat([np.repeat(z, node_num, axis=1).reshape(node_num * node_num, -1), np.repeat(z, node_num, axis=0)], 1).reshape(node_num, -1, 2*self.output_dim)
        # zizj = tf.concat(
        #     [tf.tile(z, [1, node_num * node_num]).reshape(node_num * node_num, -1),
        #      tf.tile(z, [node_num * node_num, 1])],
        #     1).reshape(node_num, -1, 2 * self.output_dim)

        c1 = tf.reshape(tf.tile(z, [1, self.coarsen_dim]), [node_num * self.coarsen_dim, -1])
        c2 = tf.tile(z, [self.coarsen_dim, 1])
        concat = tf.concat([c1, c2], 1)
        # zizj = tf.reshape(tf.concat([c1, c2], 1), [node_num, -1, 2*self.output_dim])
        zizj = tf.reshape(concat, [-1, 2*self.coarsen_dim])

        # e = tf.nn.leaky_relu(tf.reshape(tf.squeeze(dot(zizj, self.vars['a'], sparse=False)), [node_num, self.coarsen_dim]))
        e = tf.nn.leaky_relu(tf.reshape(tf.squeeze(
            dot(tf.reshape(zizj, [node_num * self.coarsen_dim, 2 * self.coarsen_dim]), self.vars['a'], sparse=False),
            axis=1), [node_num, self.coarsen_dim]))

        s = tf.nn.softmax(e)
        if self.dropout != 0:
            s = tf.nn.dropout(s, 1 - self.dropout)

        if self.bias:
            s += self.vars['bias']

        x = dot(tf.transpose(self.act(s)), x, sparse=self.sparse_inputs)
        adj = dot(dot(tf.transpose(self.act(s)), adj, sparse=False), self.act(s), sparse=False)

        return x, adj, s

class GraphConvolutionCoarsen(Layer):
    def __init__(self, input_dim, coarsen_dim, dropout, sparse_inputs, act, bias, featureless, num_supports,
                    **kwargs):
        super(GraphConvolutionCoarsen, self).__init__(**kwargs)
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.support = None
        self.act = act
        self.coarsen_dim = coarsen_dim

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_supports):
                self.vars['weights_' + str(i)] = glorot([input_dim, 1], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([1], name='bias')
        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            rt = []
            for input in inputs:
                assert (len(input) == 2)
                rt.append(self._call_one_graph(input))
            return rt
        else:
            assert (len(inputs) == 2)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, input):
        x = input[0]
        # adj = input[1]
        # num_features_nonzero = input[2]
        self.laplacians = input[1]

        x = tf.nn.dropout(x, 1 - self.dropout)

        support_list = []
        for i in range(len(self.laplacians)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_]' + str(i)]
            support = dot(self.laplacians[i], pre_sup, sparse=True)
            support_list.append(support)
        assign = tf.add_n(support_list)  # the elements of the list add one by one

        if self.bias:
            assign += self.vars['bias']

            assign = self.act(assign)
        new_x = dot(tf.transpose(assign), x, sparse=self.sparse_inputs)
        # new_adj = dot(dot(tf.transpose(assign), adj, sparse=False), assign, sparse=False)
        return new_x

# class Average(Layer):
#     """ Average layer. """
#
#     def __init__(self, **kwargs):
#         super(Average, self).__init__(**kwargs)
#
#     def produce_graph_level_emb(self):
#         return True
#
#     def merge_graph_level_embs(self):
#         return False
#
#     def _call(self, inputs):
#         if type(inputs) is list:
#             rtn = []
#             for input in inputs:
#                 rtn.append(self._call_one_mat(input))
#             return rtn
#         else:
#             return self._call_one_mat(inputs)
#
#     def _call_one_mat(self, input):
#         x = input[0]  # (N, D)
#         output = tf.reshape(tf.reduce_mean(x, 0), [1, -1])  # (1, D)
#         return output


# class Euclidean(Layer):
#     def __init__(self, input_dim, output_dim, **kwargs):
#         super(Euclidean, self).__init__(**kwargs)
#         self.output_dim = output_dim
#
#     def _call(self, inputs):
#         assert (type(inputs) is list and inputs)
#         print('inputs.len:', len(inputs))
#         rt = []
#         if type(inputs[0]) is list:
#             rt.append(self._call_one_graph(inputs))
#             for i in range(0, FLAGS.batch_size):
#                 g1 = inputs[i]
#                 g2 = inputs[i+FLAGS.batch_size]
#                 g3 = inputs[i+2*FLAGS.batch_size]
#
#                 euclidean1 = tf.sqrt(tf.reduce_sum(tf.square(g1 - g2), 1))
#                 euclidean2 = tf.sqrt(tf.reduce_sum(tf.square(g1 - g3), 1))
#                 euclidean = euclidean1 - euclidean2
#                 rt.append(euclidean)
#
#             return rt
#     def _call_one_graph(self, inputs):
#         for i in range(0, FLAGS.batch_size):
#             g1 = inputs[i]
#             g2 = inputs[i+FLAGS.batch_size]
#             g3 = inputs[i+2*FLAGS.batch_size]
#
#             euclidean1 = tf.sqrt(tf.reduce_sum(tf.square(g1 - g2), 1))
#             euclidean2 = tf.sqrt(tf.reduce_sum(tf.square(g1 - g3), 1))
#             euclidean = euclidean1 - euclidean2
#
#         return euclidean


def sparse_dropout(x, keep_prob, noise_shape):
    """ Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)