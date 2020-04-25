import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def main():
    data = pd.read_csv('data/df_final.csv', sep='|', index_col='ID')
    sample_weights = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    labels = data['CLASE'].iloc[:sample_weights.shape[0], ]
    data.drop('CLASE', axis=1, inplace=True)

    basic_cols = ['CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID', 'AREA']
    rgbn_cols = [x for x in data.columns if "RGBN_PROB_" in x]
    geom_ori_cols = [x for x in data.columns if "GEOM_R" in x]
    geom_dist_cols = [x for x in data.columns if "GEOM_DIST4" in x]
    geom_prob_cols = [x for x in data.columns if "GEOM_PROB" in x]
    xy_dens_cols = [x for x in data.columns if "XY_DENS" in x]
    xy_ori_cols = ['X', 'Y']

    custom_cols = [f'CUSTOM_{i}' for i in range(4)]
    distances_cols = [f'C_DIST_{i}' for i in range(21)]

    # points = [(2.207524e9, 165.5605e6), (2.207524e9, 165.5605e6), (2.170449e9, 166.0036e6),
    #           (2.205824e9, 166.3199e6), (2.250042e9, 166.2673e6), (2.270527e9, 165.9025e6),
    #           (2.274459e9, 165.5947e6), (2.269886e9, 165.3261e6), (2.211719e9, 165.1699e6),
    #           (2.156419e9, 165.2959e6), (2.142472e9, 165.4747e6), (2.141374e9, 165.8068e6),
    #           (2.166906e9, 165.7316e6), (2.187454e9, 165.4168e6), (2.174702e9, 165.481e6),
    #           (2.202014e9, 165.5483e6), (2.215004e9, 165.4046e6), (2.196768e9, 165.4717e6),
    #           (2.236186e9, 165.4013e6), (2.220204e9, 165.4714e6), (2.219742e9, 165.8038e6)]
    #
    # distances = [np.linalg.norm(data[['X', 'Y']].values - b, axis=1) for b in points]
    #
    # distances_cols = []
    # for i, _ in enumerate(points):
    #     col = 'C_' + str(i)
    #     distances_cols.append(col)
    #     data[col] = distances[i]

    trainOVA = pd.read_csv('data/trainOVA.txt', sep='|', index_col='ID')
    trainOVA.drop('CLASE', axis=1, inplace=True)
    ova_cols = list(trainOVA.columns.values)

    data[ova_cols] = trainOVA

    mod_cols = basic_cols + geom_ori_cols + geom_dist_cols + geom_prob_cols + rgbn_cols + xy_dens_cols + custom_cols + distances_cols

    data = data.iloc[:sample_weights.shape[0], ][mod_cols]

    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    y_pred = np.empty(labels.shape, dtype=np.chararray)
    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        # model = Pipeline([('scl', StandardScaler()),
        #                   ('clf', XGBClassifier(n_jobs=-1))])
        best_params = {'subsample': 0.6, 'n_estimators': 400, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 1.5, 'colsample_bytree': 0.6}
        model = XGBClassifier(**best_params)
        # model = XGBClassifier(learning_rate=0.1, min_child_weight=1,gamma=1,subsample=0.6,colsample_bytree=0.8,
        #                       max_depth=5,n_estimators=200, n_jobs=16)

        # model = XGBClassifier(n_jobs=16)

        print('Training...')
        sample_weight = np.array(sample_weights.values)[idx_train]*labels.size
        model.fit(data.iloc[idx_train], labels[idx_train], sample_weight=sample_weight)

        print('Predicting...')
        # pred_proba = model.predict_proba(data.iloc[idx_test])
        pred = model.predict(data.iloc[idx_test])

        y_pred[idx_test] = pred

        print('Local clasification report:')
        print('Normal:')
        print(classification_report(labels[idx_test], pred, digits=4))
        print('Weigthed:')
        print(classification_report(labels[idx_test], pred, sample_weight=sample_weights.iloc[idx_test], digits=4))

    print('Global classification report:')
    print('Normal:')
    print(classification_report(labels, y_pred, digits=4))
    print('Weigthed:')
    print(classification_report(labels, y_pred, sample_weight=sample_weights, digits=4))

if __name__=='__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()