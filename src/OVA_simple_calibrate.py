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
    labels_ini = data['CLASE'].iloc[:sample_weight.shape[0], ]
    data.drop('CLASE', axis=1, inplace=True)

    basic_cols = ['CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID', 'AREA']
    rgbn_cols = [x for x in data.columns if "RGBN_PROB_" in x]
    geom_ori_cols = [x for x in data.columns if "GEOM_R" in x]
    geom_dist_cols = [x for x in data.columns if "GEOM_DIST4" in x]
    geom_prob_cols = [x for x in data.columns if "GEOM_PROB" in x]
    xy_dens_cols = [x for x in data.columns if "XY_DENS" in x]
    xy_ori_cols = ['X', 'Y']

    data['VALUE'] = data['AREA'] * data['CADASTRALQUALITYID'] * data['MAXBUILDINGFLOOR']
    data['VALUE2'] = data['AREA'] * data['CADASTRALQUALITYID']
    data['VALUE3'] = data['CADASTRALQUALITYID'] * data['MAXBUILDINGFLOOR']
    data['VALUE4'] = data['AREA'] * data['MAXBUILDINGFLOOR']
    mod_cols = ['VALUE', 'VALUE2', 'VALUE3', 'VALUE4']

    mod_cols = basic_cols + geom_ori_cols + geom_dist_cols + geom_prob_cols + rgbn_cols + xy_dens_cols + mod_cols

    data = data.iloc[:sample_weight.shape[0],][mod_cols]

    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    labels_names = np.unique(labels_ini)

    class_weights = {'RESIDENTIAL': 4.812552140340716e-06*(labels_ini=='RESIDENTIAL').sum(),
                     'INDUSTRIAL': 4.647398736012043e-05*(labels_ini=='INDUSTRIAL').sum(),
                     'PUBLIC': 3.783937948148589e-05*(labels_ini=='PUBLIC').sum(),
                     'OFFICE': 4.558736182249404e-05*(labels_ini=='OFFICE').sum(),
                     'RETAIL': 4.2627096025849134e-05*(labels_ini=='RETAIL').sum(),
                     'AGRICULTURE': 6.261938403426534e-05*(labels_ini=='AGRICULTURE').sum(),
                     'OTHER': 3.8319803354362536e-05*(labels_ini=='OTHER').sum()}

    # values = np.unique(labels_ini, return_counts=True)
    # class_weights = {'RESIDENTIAL': 1/values[1][np.where(values[0]=='RESIDENTIAL')[0][0]]*labels_ini.size,
    #                  'INDUSTRIAL': 1/values[1][np.where(values[0]=='INDUSTRIAL')[0][0]]*labels_ini.size,
    #                  'PUBLIC': 1/values[1][np.where(values[0]=='PUBLIC')[0][0]]*labels_ini.size,
    #                  'OFFICE': 1/values[1][np.where(values[0]=='OFFICE')[0][0]]*labels_ini.size,
    #                  'RETAIL': 1/values[1][np.where(values[0]=='RETAIL')[0][0]]*labels_ini.size,
    #                  'AGRICULTURE': 1/values[1][np.where(values[0]=='AGRICULTURE')[0][0]]*labels_ini.size,
    #                  'OTHER': 1/values[1][np.where(values[0]=='OTHER')[0][0]]*labels_ini.size}

    y_pred_label_bin = {'RESIDENTIAL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'INDUSTRIAL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'PUBLIC': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'OFFICE': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'RETAIL': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'AGRICULTURE': np.ones(labels_ini.shape[0], dtype=np.int) * -2,
                        'OTHER': np.ones(labels_ini.shape[0], dtype=np.int) * -2}

    out = {}
    test = np.ones(labels_ini.shape)*-1
    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        for label in labels_names:
            print('Load %s model:' % label)

            if label != 'RESIDENTIAL':
                labels = np.array([1 if x == label else -1 for x in labels_ini])
            else:
                labels = np.array([-1 if x == label else 1 for x in labels_ini])

            class_weight = {}
            if label == 'RESIDENTIAL':
                class_weight[-1] = class_weights['RESIDENTIAL']/(labels==-1).sum()*labels_ini.size
                class_weight[1] = (1-class_weights['RESIDENTIAL'])/(labels==1).sum()*labels_ini.size
            else:
                class_weight[1] = class_weights[label]/ (labels==1).sum()*labels_ini.size
                class_weight[-1] = (1 - class_weights[label]) / (labels==-1).sum()*labels_ini.size

            # values = np.unique(labels, return_counts=True)
            # class_weight = {-1: 1/values[1][np.where(values[0]==-1)[0][0]]*labels_ini.size,
            #                 1: 1/values[1][np.where(values[0]==1)[0][0]]*labels_ini.size}
            sample_weights_bin_train = np.array([class_weight[i] for i in labels[idx_train]])
            sample_weights_bin_test = np.array([class_weight[i] for i in labels[idx_test]])

            # class_weight = {-1: (sample_weights_bin_train[labels[idx_train]==-1].sum()+sample_weights_bin_test[labels[idx_test]==-1].sum()),
            #                 1: (sample_weights_bin_train[labels[idx_train]==1].sum()+sample_weights_bin_test[labels[idx_test]==1].sum())}
            model = Pipeline([('scl', StandardScaler()),
                              ('clf', XGBClassifier(n_jobs=-1))])

            print('Training...')
            model.fit(data.iloc[idx_train], labels[idx_train], clf__sample_weight=sample_weights_bin_train)

            print('Predicting...')
            pred_proba = model.predict_proba(data)

            name = 'CV_' + str(i) + '_' + label
            if label != 'RESIDENTIAL':
                out[name] = pred_proba[:, 1]
            else:
                out[name] = pred_proba[:, 0]

        test[idx_test] = i

    out['idx_test'] = test
    out = pd.DataFrame(out, index=data.index)
    out = pd.concat([pd.DataFrame(out, index=data.index), labels_ini], axis=1, sort=False)
    out.to_csv('data/probs_ova_dg_samplew.txt', sep='|')


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
