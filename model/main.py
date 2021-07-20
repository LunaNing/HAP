# -*- coding: utf-8 -*-
import os
from time import time
import networkx as nx
import numpy as np
from data_model import load_classification_data, separate_data

import tensorflow as tf
from config import FLAGS
from model_factory import create_model
from saver import Saver
from train import train_val_loop, test
from utils_model import check_flags, get_model_info_as_str, convert_long_time_to_str, save


def main():
    t = time()
    check_flags()
    print(get_model_info_as_str())
    graphs, num_classes = load_classification_data(FLAGS.dataset, FLAGS.degree_as_tag)
    test_accs = []

    print('fold ', FLAGS.fold)
    train_graphs, test_graphs = separate_data(graphs, FLAGS.fold)
    input_dim = graphs[0].node_features.shape[1]
    model = create_model(FLAGS.model, input_dim, num_classes)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    saver = Saver(session)
    session.run(tf.global_variables_initializer())


    # train_costs, train_times, val_results_dict = train_val_loop(train_graphs, test_graphs, model, saver, session)
    train_costs, train_times, best_test_acc, best_test_auc = train_val_loop(train_graphs, test_graphs, model, saver, session)
    # test_acc, _, _= test(train_graphs, test_graphs, model, saver, session, val_results_dict)
    overall_time1 = convert_long_time_to_str(time() - t)
    print(overall_time1)
    # saver.save_overall_time(overall_time1)
    # return train_costs, train_times, val_results_dict, best_iter, test_results
    #     test_accs.append(test_acc)
    # return train_costs, train_times, val_results_dict, test_acc


if __name__ == '__main__':
    main()

