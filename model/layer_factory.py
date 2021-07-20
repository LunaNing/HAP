# -*- coding: utf-8 -*-
import tensorflow as tf
from config import FLAGS
from layer import NTN, GraphConvolution, GraphAttention, Coarsening, Average,Attention, GraphConvolutionTopk,GraphConvolutionCoarsen, Dense

def create_layers(model):
    layers = []
    num_layers = FLAGS.layer_num
    for i in range(1, num_layers + 1):
        sp = FLAGS.flag_values_dict()['layer_{}'.format(i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'GraphAttention':
            layers.append(create_GraphAttention_layer(layer_info, model, i))
        elif name == 'Coarsening':
            layers.append(create_Coarsen_layer(layer_info, model, i))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'Attention':
            layers.append(create_Attention_layer(layer_info))
        elif name == 'GraphConvolution':
            layers.append(create_GraphConvolution_layer(layer_info, model, i))
        elif name == 'GraphConvolutionTopk':
            layers.append(create_GraphConvolutionTopk_layer(layer_info, model, i))
        elif name == 'GraphConvolutionCoarsen':
            layers.append(create_GraphConvolutionCoarsen_layer(layer_info, model, i))
        elif name == 'Dense':
            layers.append(create_Dense_layer(layer_info))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info))
        # elif name == 'Euclidean':
        #     layers.append(create_Euclidean_layer(layer_info, model, i))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers

def create_NTN_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('NTN layer must have 6 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        bias=parse_as_bool(layer_info['bias']))

def create_Dense_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Dense layer must have 5 specs')
    return Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))

def create_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)

def create_GraphAttention_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphAttention layer information must have 5-6 items')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer {} must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphAttention(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
    )


# def create_Coarsen_layer(layer_info, model, layer_id):
#     input_dim = layer_info.get('input_dim')
#     if not input_dim:
#         if layer_id != 1:
#             raise RuntimeError(
#                 'The input dim for layer {} must be specified'.format(layer_id))
#         input_dim = model.input_dim
#     else:
#         input_dim = int(input_dim)
#     return Coarsening(input_dim=input_dim,
#                     coarsen_dim=int(layer_info['coarsen_dim']),
#                     temperature=float(layer_info['tem']),
#                     dropout=parse_as_bool(layer_info['dropout']),
#                     act=create_activation(layer_info['act']),
#                     bias=parse_as_bool(layer_info['bias'])
#                     )
def create_Coarsen_layer(layer_info, model, layer_id):
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer {} must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return Coarsening(input_dim=input_dim,
                    coarsen_dim=int(layer_info['coarsen_dim']),
                    dropout=parse_as_bool(layer_info['dropout']),
                    act=create_activation(layer_info['act']),
                    bias=parse_as_bool(layer_info['bias'])
                    )

def create_Average_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_Attention_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Attention layer must have 5 specs')
    return Attention(input_dim=int(layer_info['input_dim']),
                     att_times=int(layer_info['att_times']),
                     att_num=int(layer_info['att_num']),
                     att_style=layer_info['att_style'],
                     att_weight=parse_as_bool(layer_info['att_weight']))

def create_GraphConvolutionTopk_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolutionTopk layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolutionTopk(
        input_dim=input_dim,
        coarsen_dim=int(layer_info['coarsen_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)

def create_GraphConvolutionCoarsen_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolutionCoarsen layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolutionCoarsen(
        input_dim=input_dim,
        coarsen_dim=int(layer_info['coarsen_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)
# def create_Euclidean_layer(layer_info, model, layer_id):
#     if not len(layer_info) == 2:
#         raise RuntimeError('Euclidean layer information must have 2 items')
#     input_dim = layer_info.get('input_dim')
#     if not input_dim:
#         if layer_id != 1:
#             raise RuntimeError(
#                 'The input dim for layer {} must be specified'.format(layer_id))
#         input_dim = model.input_dim
#     else:
#         input_dim = int(input_dim)
#     return Euclidean(input_dim=input_dim,
#                      output_dim=int(layer_info['output_dim'])
#                      )

def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string{}'.format(b))


def create_activation(act):
    if act == 'elu':
        return tf.nn.elu
    elif act == 'relu':
        return tf.nn.relu
    elif act == 'identity':
        return tf.identity
    elif act == 'sigmoid':
        return tf.sigmoid
    elif act == 'softmax':
        return tf.nn.softmax
    elif act == 'tanh':
        return tf.tanh
    else:
        raise RuntimeError('Unknown activation function{}'.format(act))
