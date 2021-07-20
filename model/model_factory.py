# -*- coding: utf-8 -*-
# from model_regression import RegressionModel
from model_classification import ClassificationModel


def create_model(model, input_dim, num_classes):
    # if model == 'regression':
    #     return RegressionModel(input_dim, graphs)
    if model == 'classification':
        return ClassificationModel(input_dim, num_classes)
    else:
        raise RuntimeError('Unknown model {}'.format(model))
