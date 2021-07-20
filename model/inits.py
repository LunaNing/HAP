# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def xavier(shape, name):
    initial = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return initial


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    shapee=int(shape[0]) + int(shape[1])
    init_range = np.sqrt(6.0 / (shapee))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def uniform(shape, scale=0.05, name=None):
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
