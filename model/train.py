from collections import OrderedDict
from time import time
from utils_model import save
import tensorflow as tf
from sklearn.metrics import roc_auc_score


import numpy as np

from config import FLAGS
from utils_model import convert_msec_to_sec_str, get_model_info_as_str, need_val


def train_val_loop(train_graphs, test_graphs, model, saver, sess):
    train_costs, train_times, test_results_dict = [], [], OrderedDict()
    print('Optimization Started!')
    best_test_acc = 0
    best_test_auc = 0

    for iter in range(FLAGS.iters):
        iter += 1
        # Train.
        feed_dict = model.get_feed_dict_for_train(train_graphs)
        print('feed for train done!')
        train_cost, train_time = run_tf(
            feed_dict, model, saver, sess, 'train', iter=iter)

        train_costs.append(train_cost)
        train_times.append(train_time)

        #Validate.
        val_result = ''
        if need_val(iter):
            t = time()
            val_acc, val_auc, test_acc, test_auc, val_loss = val(train_graphs, test_graphs, model, saver, sess, iter)
            val_time = time() - t
            val_result += ' val_acc={}, val_auc={}, val_loss={}, test_acc={}, test_auc={}'.format(val_acc,val_auc, val_loss, test_acc,test_auc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # best_test_auc = test_auc
            if test_auc > best_test_auc:
                best_test_auc = test_auc

            # test_results_dict[iter] = test_acc
            # model.save(sess, saver, iter)


        # print('Iter:{:04n} train_loss={:.5f} train_time={} {}'.format(
        #     iter, train_cost, convert_msec_to_sec_str(train_time), val_result))


        print('iter:{:04n} train_loss={:.5f} train_time={} {}'.format(
                iter, train_cost, convert_msec_to_sec_str(train_time), val_result))



        print('best acc:', best_test_acc)
        print('best auc:', best_test_auc)

    with open('acc_result.txt', 'a+') as f:
        # f.write(str(test_loss[1]) + '\n')
        f.write(str(best_test_acc) + '\n')
    with open('auc_result.txt', 'a+') as f:
        # f.write(str(test_loss[1]) + '\n')
        f.write(str(best_test_auc) + '\n')


    print('Optimization Finished!')
    # saver.save_train_val_info(train_costs, train_times, test_results_dict)
    # return train_costs, train_times, val_results_dict
    return train_costs, train_times, best_test_acc, best_test_auc

def val(train_graphs, test_graphs, model, saver, sess, iter):
    print('validation start')
    val_acc, val_auc, test_acc, test_auc, pred_label, loss_list, time_list = run_for_val_test(
        train_graphs, test_graphs, model, saver, sess)
    val_loss = np.mean(loss_list)
    # val_results, val_s = eval.eval_for_val(pred_label, loss_list, time_list,
    #                                        model.get_eval_metrics_for_val())
    # saver.log_val_results(val_results, iter)
    return val_acc, val_auc, test_acc, test_auc, val_loss  # TODO: tensorboard


def test(train_graphs, test_graphs, model, saver, sess, val_results_dict):
    # best_iter = model.find_load_best_model(sess, saver, val_results_dict)
    # saver.clean_up_saved_models(best_iter)
    # test_acc, pred_label, time_list, graph_emb_mat, graph_label_list, orig_adj_list, targ_adj_list, att_mat_list = run_for_val_test(
    #     train_graphs, test_graphs, model, saver, sess, 'test')
    test_acc, pred_label, time_list, graph_emb_mat, graph_label_list = run_for_val_test(
        train_graphs, test_graphs, model, saver, sess, 'test')
    # node_embs_list, graph_embs_mat, emb_time = collect_embeddings(
    #     gs1, gs2, model, saver, sess)
    # attentions = collect_attentions(gs1, gs2, model, saver, sess)
    # print('Evaluating...')
    # print('test acc:', test_acc)
    # return test_acc, graph_emb_mat, graph_label_list, orig_adj_list, targ_adj_list, att_mat_list
    return test_acc, graph_emb_mat, graph_label_list
    # results = eval.eval_for_test(
    #     pred_label, loss_list, time_list,
    #     model.get_eval_metrics_for_test(), saver)
    # if not FLAGS.plot_results:
    #     pretty_print_dict(results)
    # print('Results generated with {} metrics; collecting embeddings'.format(
    #     len(results)))


def run_for_val_test(train_graphs, test_graphs, model, saver,
                             sess):
    len_val = len(train_graphs)
    len_test = len(test_graphs)
    val_pred = []
    test_pred = []
    val_true = []
    test_true = []
    val_time_list = []
    test_time_list = []
    val_loss_list = []
    test_loss_list = []
    print_count = 0
    flush = True
    val_correct = 0
    test_correct = 0
    val_scores = []
    test_scores = []
    for i in range(len_val):
        val_true_label = train_graphs[i].label
        val_true.append(val_true_label)
        feed_dict = model.get_feed_dict_for_val_test(train_graphs[i], val_true_label)
        (loss, pred), time = run_tf(feed_dict, model, saver, sess, 'val')
        if flush:
            (loss, pred), time = run_tf(feed_dict, model, saver, sess, 'val')
            flush = False
        time *= 1000
                # if val_or_test == 'test' and print_count < 100:
        val_pred.append(pred)
        val_loss_list.append(loss)
        val_time_list.append(time)
        if val_true_label == pred:
            val_correct += 1

    val_acc = val_correct /len(train_graphs)
    if model.num_class == 2:
        val_auc = roc_auc_score(val_true, val_pred)
    else:
        val_auc = 0

    # val_true = np.array(val_true)
    # val_pred = np.array(val_pred)
    # val_auc, _ = tf.metrics.auc(val_true, val_pred)
    # val_auc = tf.to_float(val_auc)


    for i in range(len_test):
        test_true_label = test_graphs[i].label
        test_true.append(test_true_label)
        feed_dict = model.get_feed_dict_for_val_test(test_graphs[i], test_true_label)
        # (orig_adj, targ_adj, att_mat, pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
        (pred), time = run_tf(feed_dict, model, saver, sess, 'test')
        if flush:
            # (orig_adj, targ_adj, att_mat, pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
            (pred), time = run_tf(feed_dict, model, saver, sess, 'test')
            flush = False
        time *= 1000

        test_pred.append(pred)
        test_time_list.append(time)
        if test_true_label == pred:
            test_correct += 1

    test_acc = test_correct/len(test_graphs)
    if model.num_class == 2:
        test_auc = roc_auc_score(test_true, test_pred)
    else:
        test_auc = 0

    # test_true = np.array(test_true)
    # test_pred = np.array(test_pred)
    # test_auc, _ = tf.metrics.auc(test_true, test_pred)
    # test_auc = tf.to_float(test_auc)
    return val_acc, val_auc, test_acc, test_auc, val_pred, val_loss_list, val_time_list
    # elif val_or_test =='test':
    #     graph_emb_mat = np.zeros((len(test_graphs), FLAGS.graph_emb))
    #     # graph_emb_mat1 = np.zeros((15, FLAGS.graph_emb1), dtype=np.int)
    #     # graph_emb_mat2 = np.zeros((7, FLAGS.graph_emb2), dtype=np.int)
    #     # graph_emb_mat3 = np.zeros((1, FLAGS.graph_emb3), dtype=np.int)
    #     graph_label_list = []
    #     orig_adj_list = []
    #     targ_adj_list = []
    #     att_mat_list = []
    #     correct = 0
    #     for i in range(len_test):
    #         true_label = test_graphs[i].label
    #         graph_label_list.append(true_label)
    #         test_true.append(true_label)
    #         feed_dict = model.get_feed_dict_for_val_test(test_graphs[i], true_label)
    #         # (orig_adj, targ_adj, att_mat, pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
    #         (pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
    #         if flush:
    #             # (orig_adj, targ_adj, att_mat, pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
    #             (pred), time = run_tf(feed_dict, model, saver, sess, val_or_test)
    #             flush = False
    #         time *= 1000
    #                 # if val_or_test == 'test' and print_count < 100:
    #         # if print_count < 100:
    #         #     print('{:.2f}mec,{:.4f},{:.4f}'.format(
    #         #         time, pred, true_label))
    #         #     print_count += 1
    #         pred_test.append(pred)
    #         # orig_adj_list.append(orig_adj)
    #         # targ_adj_list.append(targ_adj)
    #         # att_mat_list.append(att_mat)
    #
    #
    #         # graph_emb_mat[i] = graph_embedding[0]
    #
    #         test_time_list.append(time)
    #         if true_label == pred:
    #             correct += 1
    #     test_acc = correct/len(test_graphs)
        # print('graph embedding mat:', graph_emb_mat)
        # return test_acc, pred_test, test_time_list, graph_emb_mat, graph_label_list, orig_adj_list, targ_adj_list, att_mat_list
        # return test_acc, pred_test, test_time_list, graph_emb_mat, graph_label_list


def run_tf(feed_dict, model, saver, sess, tvt, iter=None):
    if tvt == 'train':
        objs = [model.opt_op, model.train_loss]
    elif tvt == 'val':
        objs = [model.val_test_loss, model.pred_sim_without_act()]
    elif tvt == 'test':
        # objs = [model.orig_adj, model.targ_adj, model.att_mat, model.pred_sim_without_act()]
        objs = [model.pred_sim_without_act()]
    # elif tvt == 'test_emb':
    #     objs = [model.node_embeddings, model.graph_embeddings]
    # elif tvt == 'test_att':
    #     objs = [model.attentions]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt, iter)
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'train':
        # print('train act:', act)
        rtn = outs[-1]
    elif tvt == 'val' or tvt == 'test':
        # np_result = model.apply_final_act_np(outs[-1])
        np_result = outs[-1]
        if tvt == 'val':
            rtn = (outs[-2], np_result)
        else:
            rtn = (np_result)
        # rtn = (outs[-2], np_result)
    elif tvt == 'test_emb':
        rtn = (outs[-2], outs[-1])
    else:
        rtn = outs[-1]
    return rtn, time_rtn

