# -*- coding: utf-8 -*-

import tensorflow as tf

flags = tf.app.flags

dataset = 'COLLAB'
flags.DEFINE_string('dataset', dataset, 'Dataset String')
flags.DEFINE_boolean('degree_as_tag', True, 'degree as tag')

# if 'aids' in dataset:
#     node_feat_name = 'type'
#     node_feat_encoder = 'onehot'
#     max_nodes = 10
# elif dataset == 'linux' or 'imdb' in dataset:
#     node_feat_name = None
#     node_feat_encoder = 'constant_1'  # 1 means the input_dim of constant nodes
#     if dataset == 'linux':
#         max_nodes = 10
#     else:
#         max_nodes = 90
# else:
#     max_nodes = 10000
#     # assert (False)

flags.DEFINE_integer('gpu', 0, 'Which gpu to use')

"""model:regression or classification"""
model = 'classification'
flags.DEFINE_string('model', model, 'Model string.')

flags.DEFINE_boolean('random_permute', False,
                     'Whether to random permute nodes of graphs in training or not.')
flags.DEFINE_integer('graph_emb', 32, 'The dimension of graph level embedding')
# flags.DEFINE_integer('graph_emb2', 32, 'The dimension of graph level embedding')
# flags.DEFINE_integer('graph_emb3', 16, 'The dimension of graph level embedding')
flags.DEFINE_integer('fold',1, 'fold(1..10)')
flags.DEFINE_integer('batch_size', 16, 'Number of graph pairs in a batch')#128 for classification
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')#0.01 for classification
flags.DEFINE_float('dropout', 0.5, 'Dropout rate')
# flags.DEFINE_float('valid_percentage', 0.25,
#                    'The percentage of validation graphs in the sum of validation graphs and train graphs')
flags.DEFINE_integer('iters', 300 , 'Number of iterations to train')#1000 for classification
flags.DEFINE_integer('iters_val_start', 1, 'Which iteration validation start')
flags.DEFINE_integer('iters_val_every',1, 'Frequency of validation,valid every 2 iters')

if model == 'classification':
    flags.DEFINE_integer('layer_num', 5, 'Number of layers')
    flags.DEFINE_string('layer_1',
                        'GraphAttention:output_dim=64,dropout=False,bias=True,act=relu,sparse_inputs=False',
                        '')
    flags.DEFINE_string('layer_2',
                        'GraphAttention:input_dim=64,output_dim=32,dropout=False,bias=True,act=relu,sparse_inputs=False',
                        '')
    # # flags.DEFINE_string('layer_3',
    # #                     'GraphAttention:input_dim=32,output_dim=16,dropout=False,bias=True,act=relu,sparse_inputs=False',
    # #                     '')
    # flags.DEFINE_string(
    #     'layer_3',
    #     'Average', '')
    # flags.DEFINE_string(
    #     'layer_3',
    #     'Attention:input_dim=32,att_times=1,att_num=1,att_weight=True,att_style=dot', '')
    # flags.DEFINE_string('layer_3',
    #                         'GraphConvolutionTopk:input_dim=32,coarsen_dim=1,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                         '')
    flags.DEFINE_string(
        'layer_3',
        'GraphConvolutionCoarsen:input_dim=32,coarsen_dim=1,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False', '')
    flags.DEFINE_string(
        'layer_4',
        'Dense:input_dim=32,output_dim=64,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_5',
        'Dense:input_dim=64,output_dim=3,dropout=True,bias=True,'
        'act=sigmoid', '')
    # flags.DEFINE_string('layer_1',
    #                     'GraphAttention:output_dim=128,dropout=False,bias=False,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_2',
    #                     'GraphAttention:input_dim=128,output_dim=64,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_3',
    #                     'Coarsening:input_dim=64,coarsen_dim=3,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_4',
    #                     'GraphAttention:input_dim=64,output_dim=32,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_5',
    #                     'Coarsening:input_dim=32,coarsen_dim=1,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')


    # flags.DEFINE_string('layer_6',
    #                     'GraphAttention:input_dim=64,output_dim=32,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_7',
    #                     'Coarsening:input_dim=32,coarsen_dim=1,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')



    # flags.DEFINE_string(
    #     'layer_6',
    #     'Dense:input_dim=32,output_dim=64,dropout=False,bias=True,'
    #     'act=relu', '')
    # flags.DEFINE_string(
    #     'layer_7',
    #     'Dense:input_dim=64,output_dim=3,dropout=False,bias=True,'
    #     'act=sigmoid', '')



    # flags.DEFINE_string('layer_1',
    #                     'GraphAttention:output_dim=64,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string(
    #     'layer_2',
    #     'GraphAttention:input_dim=64,output_dim=32,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # # flags.DEFINE_string(
    # #     'layer_3',
    # #     'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
    # #     'act=identity,sparse_inputs=False', '')
    # flags.DEFINE_string('layer_3',
    #                     'Coarsening:input_dim=32,coarsen_dim=1,tem=0.1,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=relu', '')
    # flags.DEFINE_string(
    #     'layer_5',
    #     'Dense:input_dim=16,output_dim=2,dropout=True,bias=True,'
    #     'act=softmax', '')

    # flags.DEFINE_string(
    #     'layer_4',
    #     'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # flags.DEFINE_string('layer_5',
    #                     'GraphAttention:input_dim=32,output_dim=16,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                     '')
    # flags.DEFINE_string('layer_7',
    #                         'GraphAttention:input_dim=8,output_dim=4,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                         '')
    # flags.DEFINE_string('layer_8',
    #                         'GraphAttention:input_dim=16,output_dim=16,dropout=False,bias=True,act=relu,sparse_inputs=False',
    #                         '')

    # flags.DEFINE_string(
    #     'layer_6',
    #     'NTN:input_dim=32,feature_map_dim=32,dropout=False,bias=True,'
    #     'inneract=relu,apply_u=False', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=relu', '')
    # flags.DEFINE_string(
    #     'layer_5',
    #     'Dense:input_dim=16,output_dim=3,dropout=True,bias=True,'
    #     'act=relu', '')


    flags.DEFINE_float('weight_decay', 0.00001,
                       'Weight for L2 loss on embedding matrix.')

    thresh = 0

    flags.DEFINE_float('thresh_pos', thresh,
                       'Threshold below which train pairs are similar.')
    flags.DEFINE_float('thresh_neg', thresh,
                       'Threshold above which train pairs are dissimilar.')
    flags.DEFINE_float('thresh_val_test_pos', thresh,
                       'Threshold that binarizes test pairs.')
    flags.DEFINE_float('thresh_val_test_neg', thresh,
                       'Threshold that binarizes test pairs.')

flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results '
                     '(involving all baselines) or not.')
FLAGS = tf.app.flags.FLAGS
