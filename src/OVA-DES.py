import os
import pickle

import numpy as np
import pandas as pd
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main():
    data = pd.read_csv('data/train.txt', sep='|', index_col='ID').iloc[:1000, ]
    labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    # data, test, labels_ini, y_test = train_test_split(data, labels_ini, test_size=0.9, random_state=42)

    weights = pd.read_csv('data/train_weights.cvs.txt', sep='|', index_col='ID').iloc[:1000, ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    labels_names = np.unique(labels_ini)
    labels_to_int = {'AGRICULTURE': 0, 'INDUSTRIAL': 1, 'OFFICE': 2, 'OTHER': 3, 'PUBLIC': 4,
                     'RESIDENTIAL': 5, 'RETAIL': 6}
    labels_int = labels_ini.map(labels_to_int)

    alpha = 0.1
    k = 3 * 7

    class_weights = {'RESIDENTIAL': 4.812552140340716e-06,
                     'INDUSTRIAL': 4.647398736012043e-05,
                     'PUBLIC': 3.783937948148589e-05,
                     'OFFICE': 4.558736182249404e-05,
                     'RETAIL': 4.2627096025849134e-05,
                     'AGRICULTURE': 6.261938403426534e-05,
                     'OTHER': 3.8319803354362536e-05}

    print('---------------------------------OVA-DES MODEL--------------------------------')
    y_pred = np.ones(labels_ini.shape[0], dtype=np.int) * -1
    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        y_pred_label = []
        for label in labels_names:
            print('Load %s model:' % label)

            dump_file = './models/' + label + '_best_gs_pipeline.pkl'
            with open(dump_file, 'rb') as ofile:
                grid = pickle.load(ofile)

            model = grid.best_estimator_
            for step in model.steps:
                if step[0] in ['enn', 'clf']:
                    step[1].n_jobs = -1

            if label != 'RESIDENTIAL':
                labels = np.array([1 if x == label else -1 for x in labels_ini])
            else:
                labels = np.array([-1 if x == label else 1 for x in labels_ini])

            class_weight = {}
            if label == 'RESIDENTIAL':
                class_weight[-1] = class_weights['RESIDENTIAL']
                class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
            else:
                class_weight[1] = class_weights[label]
                class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])

            sample_weights_bin = np.array([class_weight[i] for i in labels[idx_test]])

            print('Training...')
            model.fit(data.iloc[idx_train], labels[idx_train])

            print('Predicting...')
            pred_proba = model.predict_proba(data.iloc[idx_test])
            pred = model.predict(data.iloc[idx_test])

            print('Local clasification report:')
            print('Normal')
            print(classification_report(labels[idx_test], pred))
            print('Weigthed:')
            print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin))

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])

        ############################# DES ###########################
        # Dynamic ensemble selection for multi-class classification with one-class classifiers
        y_pred_label = np.array(y_pred_label).T

        knn = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='manhattan')
        knn.fit(data.iloc[idx_train], labels_ini.iloc[idx_train])
        neighbors = knn.kneighbors(data.iloc[idx_test])[1]

        labels_neigh = [np.unique(labels_int.loc[data.iloc[idx_train].iloc[neighs].index]) for neighs in neighbors]

        r_idx,c_idx = list(map(np.concatenate,zip(*
                 [([i]*(7-len(x)),                  # i*numero de veces
                   np.setdiff1d(np.arange(7),x))    # [1..7]-x
                  for i,x in enumerate(labels_neigh)])))
        r_idx = r_idx.astype(np.int)
        c_idx = c_idx.astype(np.int)
        if len(r_idx) > 0 and len(c_idx) > 0:
            y_pred_label[r_idx,c_idx] = 0

        #############################################################

        ############################# DESthr ###########################
        # Dynamic ensemble selection for multi-class classification with one-class classifiers
        # y_pred_label = np.array(y_pred_label).T
        #
        # knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        # knn.fit(data.iloc[idx_train], labels_ini.iloc[idx_train])
        # neighbors = knn.kneighbors(data.iloc[idx_test])[1]

        # TODO se elenan las clases que tiene menos de alpha*k elementos
        labels_neigh = [np.unique(labels_int.loc[data.iloc[idx_train].iloc[neighs].index]) for neighs in neighbors]

        r_idx, c_idx = list(map(np.concatenate, zip(*
                                                    [([i] * (7 - len(x)),  # i*numero de veces
                                                      np.setdiff1d(np.arange(7), x))  # [1..7]-x
                                                     for i, x in enumerate(labels_neigh)])))
        r_idx = r_idx.astype(np.int)
        c_idx = c_idx.astype(np.int)
        if len(r_idx) > 0 and len(c_idx) > 0:
            y_pred_label[r_idx, c_idx] = 0

        #############################################################

        y_pred[idx_test] = np.argmax(y_pred_label, axis=1)

        print('Fold classification report:')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]]))

    assert -1 not in np.unique(y_pred)

    print('Global classification report:')
    print(classification_report(labels_ini, labels_names[y_pred]))

    n_train = pd.DataFrame(y_pred_label, index=data.index, columns=labels_names)
    pd.concat([y_pred, labels_ini], axis=1).to_csv('data/trainOVA-DES.txt', sep='|')


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()