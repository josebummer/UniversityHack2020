import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle

import numpy as np
import pandas as pd
import progressbar
from sklearn.model_selection import KFold

from expermientos1 import  prepare_data, fillna, to_numeric

import json

#
# CONFIG
#

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label','-l',dest='label',required=True)
parser.add_argument('--add','-a',dest='to_add', type=str, nargs='+', default=[])
parser.add_argument('--drop','-d',dest='to_drop', type=str, nargs='+', default=[])
args = parser.parse_args()

# OUT_LABEL = "GEOM"
# cols_to_select_label = ["GEOM_"]
# cols_to_drop_label = ["GEOM_"]

OUT_LABEL = args.label
cols_to_select_label = args.to_add
cols_to_drop_label = args.to_drop


def main():
    data = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    data_nn = pd.read_csv('data/train_NN.txt', sep='|', index_col='ID')

    data_ts = pd.read_csv('data/test.txt', sep='|', index_col='ID')
    data_nn_ts = pd.read_csv('data/test_NN.txt', sep='|', index_col='ID')

    cv_idx = data_nn.cv_idx.to_numpy()
    folds = [[np.where(cv_idx != i)[0], np.where(cv_idx == i)[0]] for i in range(5)]

    labels_ini = data.CLASE
    data.drop(columns=['CLASE'], inplace=True)
    data.drop(columns=[k for l in cols_to_drop_label for k in data.columns if l in k], inplace=True)
    data_ts.drop(columns=[k for l in cols_to_drop_label for k in data_ts.columns if l in k], inplace=True)

    data_nn_list = [data_nn[[k for l in cols_to_select_label for k in data_nn.columns if l in k and f'VAL_{i}' in k]] for i in range(5)]
    data_nn_ts_train = data_nn[[k for l in cols_to_select_label for k in data_nn.columns if l in k and "_TOTAL" in k]]
    data_nn_ts_predict = data_nn_ts[[k for l in cols_to_select_label for k in data_nn_ts.columns if l in k]]

    def transf_data(x):
        x = prepare_data(x)
        x = fillna(x)
        x = to_numeric(x)
        return x

    data_cv = [transf_data(pd.concat([data,data_nn_cv],axis=1)) for data_nn_cv in data_nn_list]
    data_ts_train = transf_data(pd.concat([data,data_nn_ts_train],axis=1))
    data_ts_predict = transf_data(pd.concat([data_ts, data_nn_ts_predict], axis=1))

    labels_names = np.unique(labels_ini)


    print('Creating data...')
    # kf = KFold(n_splits=5, shuffle=True, random_state=37)
    # folds = list(kf.split(data))

    y_pred = np.ones((data.shape[0], len(labels_names)), dtype=np.float)*-1
    for i, (idx_train, idx_test) in enumerate(folds):
        print('Fold %d' % i)

        y_pred_label = []
        for label in progressbar.progressbar(labels_names):
            print('Load %s model:' % label)

            dump_file = './1models_nw_dn/' + label + '_best_gs_pipeline.pkl'
            with open(dump_file, 'rb') as ofile:
                grid = pickle.load(ofile)

            model = grid.best_estimator_
            for l in model.steps:
                if 'n_jobs' in l[1].get_params():
                    l[1].set_params(**{'n_jobs': -1})

            if label != 'RESIDENTIAL':
                labels = np.array([1 if x == label else -1 for x in labels_ini])
            else:
                labels = np.array([-1 if x == label else 1 for x in labels_ini])

            print('Training...')
            model.fit(data_cv[i].iloc[idx_train], labels[idx_train])

            print('Predicting...')
            pred_proba = model.predict_proba(data_cv[i].iloc[idx_test])

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])

        y_pred[idx_test] = np.array(y_pred_label).T

    n_data = pd.DataFrame(y_pred, index=data.index, columns=labels_names)

    print('Save new train')
    pd.concat([n_data, labels_ini], axis=1).to_csv(f'data/train_{OUT_LABEL}.txt', sep='|')

    #
    # TEST
    #

    y_pred_label = []
    for label in progressbar.progressbar(labels_names):
        print('Load %s model:' % label)

        dump_file = './1models_nw_dn/' + label + '_best_gs_pipeline.pkl'
        with open(dump_file, 'rb') as ofile:
            grid = pickle.load(ofile)

        model = grid.best_estimator_
        for l in model.steps:
            if 'n_jobs' in l[1].get_params():
                l[1].set_params(**{'n_jobs': -1})

        if label != 'RESIDENTIAL':
            labels = np.array([1 if x == label else -1 for x in labels_ini])
        else:
            labels = np.array([-1 if x == label else 1 for x in labels_ini])

        print('Training...')
        model.fit(data_ts_train, labels)

        print('Predicting...')
        pred_proba = model.predict_proba(data_ts_predict)

        if label != 'RESIDENTIAL':
            y_pred_label.append(pred_proba[:, 1])
        else:
            y_pred_label.append(pred_proba[:, 0])

    y_pred = np.array(y_pred_label).T

    n_data = pd.DataFrame(y_pred, index=data_ts.index, columns=labels_names)

    print('Save new test')
    n_data.to_csv(f'data/test_{OUT_LABEL}.txt', sep='|')

if __name__ == '__main__':
    main()
