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
OUT_LABEL = 'EVERY_MASK'
mask_list = []
for m in ["AGRICULTURE","INDUSTRIAL","OFFICE","OTHER","PUBLIC","RESIDENTIAL","RETAIL"]:
    with open(f'./GEN_OUTPUT_{m}.json', 'r') as ifile:
        j = json.load(ifile)
    mask = j['best_x'][np.argmin(j['best_f'])]
    mask = np.array(mask, dtype=np.bool)
    mask_list.append((m,mask))

mask_dict = dict(mask_list)
# mask_dict = dict([(m,mask) for m in ["AGRICULTURE","INDUSTRIAL","OFFICE","OTHER","PUBLIC","RESIDENTIAL","RETAIL"]])

def main():
    data = pd.read_csv('data/train.txt', sep='|', index_col='ID')

    labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    labels_names = np.unique(labels_ini)


    print('Creating data...')
    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    folds = list(kf.split(data))

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
            model.fit(data.iloc[idx_train].loc[:,mask_dict[label]], labels[idx_train])

            print('Predicting...')
            pred_proba = model.predict_proba(data.iloc[idx_test].loc[:,mask_dict[label]])

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])
            # y_pred_label.append(pred_proba[:, 1])

        y_pred[idx_test] = np.array(y_pred_label).T

    n_data = pd.DataFrame(y_pred, index=data.index, columns=labels_names)

    print('Save new train')
    pd.concat([n_data, labels_ini], axis=1).to_csv(f'data/train_mask_{OUT_LABEL}.txt', sep='|')

if __name__ == '__main__':
    main()
