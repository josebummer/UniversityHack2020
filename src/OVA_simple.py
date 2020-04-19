import os
import pickle
import json

import numpy as np
import pandas as pd
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    print('---------------------------------OVA-Simple-----------------------------------')
    data = pd.read_csv('data/df_final.csv', sep='|', index_col='ID')
    sample_weight = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    labels_ini = data['CLASE']
    data.drop('CLASE', axis=1, inplace=True)

    basic_cols = ['CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID', 'AREA']
    rgbn_cols = [x for x in data.columns if "RGBN_PROB_" in x]
    geom_ori_cols = [x for x in data.columns if "GEOM_R" in x]
    geom_dist_cols = [x for x in data.columns if "GEOM_DIST4" in x]
    geom_prob_cols = [x for x in data.columns if "GEOM_PROB" in x]
    xy_dens_cols = [x for x in data.columns if "XY_DENS" in x]
    xy_ori_cols = ['X', 'Y']

    mod_cols = basic_cols + geom_ori_cols + geom_dist_cols + geom_prob_cols + rgbn_cols + xy_dens_cols

    data = data.iloc[:sample_weight.shape[0],][mod_cols]

    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    labels_names = np.unique(labels_ini)

    # class_weights = {'RESIDENTIAL': 4.812552140340716e-06,
    #                  'INDUSTRIAL': 4.647398736012043e-05,
    #                  'PUBLIC': 3.783937948148589e-05,
    #                  'OFFICE': 4.558736182249404e-05,
    #                  'RETAIL': 4.2627096025849134e-05,
    #                  'AGRICULTURE': 6.261938403426534e-05,
    #                  'OTHER': 3.8319803354362536e-05}

    values = np.unique(labels_ini, return_counts=True)
    class_weights = {'RESIDENTIAL': 1/values[1][np.where(values[0]=='RESIDENTIAL')[0][0]],
                     'INDUSTRIAL': 1/values[1][np.where(values[0]=='INDUSTRIAL')[0][0]],
                     'PUBLIC': 1/values[1][np.where(values[0]=='PUBLIC')[0][0]],
                     'OFFICE': 1/values[1][np.where(values[0]=='OFFICE')[0][0]],
                     'RETAIL': 1/values[1][np.where(values[0]=='RETAIL')[0][0]],
                     'AGRICULTURE': 1/values[1][np.where(values[0]=='AGRICULTURE')[0][0]],
                     'OTHER': 1/values[1][np.where(values[0]=='OTHER')[0][0]]}

    y_pred_label_bin = {'RESIDENTIAL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'INDUSTRIAL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'PUBLIC': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'OFFICE': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'RETAIL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'AGRICULTURE': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'OTHER': np.ones(labels_ini.shape[0], dtype=np.int) * -2}

    y_pred = np.ones(labels_ini.shape[0], dtype=np.int) * -1
    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        y_pred_label = []
        for label in labels_names:
            print('Load %s model:' % label)

            model = Pipeline([('scl', StandardScaler()),
                              ('clf', XGBClassifier(n_jobs=-1))])

            if label != 'RESIDENTIAL':
                labels = np.array([1 if x == label else -1 for x in labels_ini])
            else:
                labels = np.array([-1 if x == label else 1 for x in labels_ini])

            # class_weight = {}
            # if label == 'RESIDENTIAL':
            #     class_weight[-1] = class_weights['RESIDENTIAL']
            #     class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
            # else:
            #     class_weight[1] = class_weights[label]
            #     class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])

            values = np.unique(labels, return_counts=True)
            class_weight = {-1: 1/values[1][np.where(values[0]==-1)[0][0]],
                            1: 1/values[1][np.where(values[0]==1)[0][0]]}
            sample_weights_bin = np.array([class_weight[i] for i in labels[idx_test]])

            print('Training...')
            model.fit(data.iloc[idx_train], labels[idx_train], sample_weight=sample_weights_bin)

            print('Predicting...')
            pred_proba = model.predict_proba(data.iloc[idx_test])
            pred = model.predict(data.iloc[idx_test])

            print('Local clasification report:')
            print('Normal:')
            print(classification_report(labels[idx_test], pred, digits=4))
            print('Weigthed:')
            print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin, digits=4))

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])

            y_pred_label_bin[label][idx_test] = pred

        y_pred[idx_test] = np.argmax(y_pred_label, axis=0)

        print('Fold classification report:')
        print('Normal')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]], digits=4))
        print('Weigthed:')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]],
                                    sample_weight=sample_weight.iloc[idx_test], digits=4))

    assert -1 not in np.unique(y_pred)

    for label in labels_names:
        if label != 'RESIDENTIAL':
            labels = np.array([1 if x == label else -1 for x in labels_ini])
        else:
            labels = np.array([-1 if x == label else 1 for x in labels_ini])

        # class_weight = {}
        # if label == 'RESIDENTIAL':
        #     class_weight[-1] = class_weights['RESIDENTIAL']
        #     class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
        # else:
        #     class_weight[1] = class_weights[label]
        #     class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
        values = np.unique(labels, return_counts=True)
        class_weight = {-1: 1 / values[1][np.where(values[0] == -1)[0][0]],
                        1: 1 / values[1][np.where(values[0] == 1)[0][0]]}
        sample_weights_bin = np.array([class_weight[i] for i in labels])

        assert -2 not in np.unique(y_pred_label_bin[label])

        print('Label binary ' + label + ' classification report:')
        print('Normal:')
        print(classification_report(labels, y_pred_label_bin[label], digits=4))
        print('Weigthed:')
        print(classification_report(labels, y_pred_label_bin[label], sample_weight=sample_weights_bin, digits=4))
        print('Confusion matrix:')
        print(confusion_matrix(labels, y_pred_label_bin[label]))

    print('Global classification report:')
    print('Normal:')
    print(classification_report(labels_ini, labels_names[y_pred], digits=4))
    print('Weigthed:')
    print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight, digits=4))


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
