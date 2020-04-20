import os
import pickle

import numpy as np
import pandas as pd
import progressbar
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

from aux_scorer import get_weight_f1


class AddColumns(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        points = [(2.207524e9, 165.5605e6), (2.207524e9, 165.5605e6), (2.170449e9, 166.0036e6),
                  (2.205824e9, 166.3199e6), (2.250042e9, 166.2673e6), (2.270527e9, 165.9025e6),
                  (2.274459e9, 165.5947e6), (2.269886e9, 165.3261e6), (2.211719e9, 165.1699e6),
                  (2.156419e9, 165.2959e6), (2.142472e9, 165.4747e6), (2.141374e9, 165.8068e6),
                  (2.166906e9, 165.7316e6), (2.187454e9, 165.4168e6), (2.174702e9, 165.481e6),
                  (2.202014e9, 165.5483e6), (2.215004e9, 165.4046e6), (2.196768e9, 165.4717e6),
                  (2.236186e9, 165.4013e6), (2.220204e9, 165.4714e6), (2.219742e9, 165.8038e6)]

        distances = [np.linalg.norm(data[['X', 'Y']].values - b, axis=1) for b in points]

        for i, _ in enumerate(points):
            col = 'C_' + str(i)
            data[col] = distances[i]

        data.drop(columns=['X', 'Y'], axis=1, inplace=True)

        return data


class DeleteXY(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        data.drop(columns=['X', 'Y'], axis=1, inplace=True)

        return data


def prepare_data(pdata):
    data = pdata.copy()
    data['CADASTRALQUALITYID'] = data['CADASTRALQUALITYID'].map({'9': '0',
                                                                 '8': '1',
                                                                 '7': '2',
                                                                 '6': '3',
                                                                 '5': '4',
                                                                 '4': '5',
                                                                 '3': '6',
                                                                 '2': '7',
                                                                 '1': '8',
                                                                 'C': '9',
                                                                 'B': '10',
                                                                 'A': '11', })
    data['CADASTRALQUALITYID'] = data['CADASTRALQUALITYID'].astype('category')

    return data


def fillna(pdata):
    data = pdata.copy()

    data['MAXBUILDINGFLOOR'].fillna(data['MAXBUILDINGFLOOR'].median(), inplace=True)
    data['CADASTRALQUALITYID'].fillna(data['CADASTRALQUALITYID'].mode()[0], inplace=True)

    return data


def to_numeric(pdata):
    data = pdata.copy()

    data['CADASTRALQUALITYID'] = data['CADASTRALQUALITYID'].astype(np.int)

    return data


def main():
    # Load and split the data
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
    product_cols = ['VALUE', 'VALUE2', 'VALUE3', 'VALUE4']

    mod_cols = basic_cols + geom_ori_cols + geom_dist_cols + geom_prob_cols + rgbn_cols + xy_dens_cols + product_cols + xy_ori_cols

    train = data.iloc[:sample_weight.shape[0], ][mod_cols]

    # class_weights = {'RESIDENTIAL': 4.812552140340716e-06*train.shape[0],
    #                 'INDUSTRIAL': 4.647398736012043e-05*train.shape[0],
    #                 'PUBLIC': 3.783937948148589e-05*train.shape[0],
    #                 'OFFICE': 4.558736182249404e-05*train.shape[0],
    #                 'RETAIL': 4.2627096025849134e-05*train.shape[0],
    #                 'AGRICULTURE': 6.261938403426534e-05*train.shape[0],
    #                 'OTHER': 3.8319803354362536e-05*train.shape[0]}
    class_weights = {'RESIDENTIAL': 4.812552140340716e-06 * (labels_ini == 'RESIDENTIAL').sum(),
                     'INDUSTRIAL': 4.647398736012043e-05 * (labels_ini == 'INDUSTRIAL').sum(),
                     'PUBLIC': 3.783937948148589e-05 * (labels_ini == 'PUBLIC').sum(),
                     'OFFICE': 4.558736182249404e-05 * (labels_ini == 'OFFICE').sum(),
                     'RETAIL': 4.2627096025849134e-05 * (labels_ini == 'RETAIL').sum(),
                     'AGRICULTURE': 6.261938403426534e-05 * (labels_ini == 'AGRICULTURE').sum(),
                     'OTHER': 3.8319803354362536e-05 * (labels_ini == 'OTHER').sum()}

    # train, test, yy_train, yy_test = train_test_split(train, labels_ini, test_size=0.2, random_state=42)

    # SI SE CALCULAN LOS MODELOS CON PESOS, RESIDENTIAL VA NORMAL, SI SE CALCULAN SIN PESOS, RESIDENTIAL INVERTIDO.
    for label in progressbar.progressbar(np.unique(labels_ini)[2:]):
        print('\n--------------------OVA: %s vs All------------------' % label)
        if label != 'RESIDENTIAL':
            labels = np.array([1 if x == label else -1 for x in labels_ini])
        else:
            labels = np.array([-1 if x == label else 1 for x in labels_ini])
        # labels = np.array([1 if x == label else -1 for x in labels_ini])

        class_weight = {}
        if label == 'RESIDENTIAL':
            class_weight[-1] = class_weights['RESIDENTIAL'] / (labels == -1).sum() * labels_ini.size
            class_weight[1] = (1 - class_weights['RESIDENTIAL']) / (labels == 1).sum() * labels_ini.size
        else:
            class_weight[1] = class_weights[label] / (labels == 1).sum() * labels_ini.size
            class_weight[-1] = (1 - class_weights[label]) / (labels == -1).sum() * labels_ini.size

        # f1w_scorer = get_weight_f1(class_weight)
        f1w = get_weight_f1(class_weight)
        f1w_scorer = make_scorer(f1w)

        X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)
        sample_weights_tr = np.array([class_weight[i] for i in y_train])
        sample_weights_tst = np.array([class_weight[i] for i in y_test])

        # Construct some pipelines
        pipe_rf = Pipeline([('rxy', DeleteXY()),
                             # ('scl', StandardScaler()),
                            ('clf', RandomForestClassifier())])

        pipe_rf_dist = Pipeline([('add', AddColumns()),
                                    # ('scl', StandardScaler()),
                                  ('clf', RandomForestClassifier())])

        pipe_knn = Pipeline([('rxy', DeleteXY()),
                             # ('scl', StandardScaler()),
                             ('clf', KNeighborsClassifier())])

        pipe_knn_dist = Pipeline([('add', AddColumns()),
                                   # ('scl', StandardScaler()),
                                   ('clf', KNeighborsClassifier())])

        pipe_xgb = Pipeline([('rxy', DeleteXY()),
                             # ('scl', StandardScaler()),
                             ('clf', XGBClassifier())])

        pipe_xgb_dist = Pipeline([('add', AddColumns()),
                                   # ('scl', StandardScaler()),
                                   ('clf', XGBClassifier())])

        # Set grid search params
        grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                           'clf__min_samples_leaf': [1, 3, 5],
                           'clf__max_depth': [None, 2, 5, 7],
                           'clf__n_estimators': [100, 400, 600, 900, 1200],
                           'clf__min_samples_split': [2, 5, 8]}]

        grid_params_knn = [{'clf__n_neighbors': [1, 3, 5, 7, 11, 13],
                            'clf__weights': ['uniform', 'distance'],
                            'clf__metric': ['minkowski', 'manhattan']}]

        grid_params_xgboost = [{'clf__max_depth': [2, 6, 8, 12, 18],
                                'clf__learning_rate': [0.1, 0.15, 0.3, 0.4, 0.5],
                                'clf__n_estimators': [400, 600, 900, 1200]}]

        # Construct grid searches
        jobs = -1


        gs_rf = RandomizedSearchCV(estimator=pipe_rf,
                                   param_distributions=grid_params_rf,
                                   scoring=f1w_scorer,
                                   cv=5,
                                   n_jobs=jobs,
                                   n_iter=20)

        gs_rf_dist = RandomizedSearchCV(estimator=pipe_rf_dist,
                                         param_distributions=grid_params_rf,
                                         scoring=f1w_scorer,
                                         cv=5,
                                         n_jobs=jobs,
                                         n_iter=20)

        gs_knn = RandomizedSearchCV(estimator=pipe_knn,
                                    param_distributions=grid_params_knn,
                                    scoring=f1w_scorer,
                                    cv=5,
                                    n_jobs=jobs,
                                    n_iter=20)

        gs_knn_dist = RandomizedSearchCV(estimator=pipe_knn_dist,
                                          param_distributions=grid_params_knn,
                                          scoring=f1w_scorer,
                                          cv=5,
                                          n_jobs=jobs,
                                          n_iter=20)

        gs_xgb = RandomizedSearchCV(estimator=pipe_xgb,
                                    param_distributions=grid_params_xgboost,
                                    scoring=f1w_scorer,
                                    cv=5,
                                    n_jobs=jobs,
                                    n_iter=20)

        gs_xgb_dist = RandomizedSearchCV(estimator=pipe_xgb_dist,
                                          param_distributions=grid_params_xgboost,
                                          scoring=f1w_scorer,
                                          cv=5,
                                          n_jobs=jobs,
                                          n_iter=20)

        # List of pipelines for ease of iteration
        grids = [gs_rf, gs_rf_dist, gs_knn, gs_knn_dist, gs_xgb, gs_xgb_dist]

        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'Random Forest', 1: 'Random Forest w/ dist',
                     2: 'KNN', 3: 'KNN w/ dist',
                     4: 'XGB', 5: 'XGB w/ dist'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_f1 = 0.0
        best_clf = 0
        best_gs = ''
        for idx in progressbar.progressbar(range(len(grids))):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            if idx != 2 and idx != 3:
                grids[idx].fit(X_train, y_train, clf__sample_weight=sample_weights_tr)
            else:
                grids[idx].fit(X_train, y_train)
            # Best params
            print('Best params: %s' % grids[idx].best_params_)
            # Best training data accuracy
            print('Best training f1: %.3f' % grids[idx].best_score_)
            # Predict on test data with best params
            y_pred = grids[idx].predict(X_test)
            # Test data accuracy of model with best params
            print('Test set metrics for best params:')
            print('Normal clasification:')
            print(classification_report(y_test, y_pred))
            print('Weighted clasification:')
            print(classification_report(y_test, y_pred, sample_weight=sample_weights_tst))
            # Track best (highest test f1) model
            if f1_score(y_test, y_pred, sample_weight=sample_weights_tst, average='weighted') > best_f1:
                best_f1 = f1_score(y_test, y_pred, sample_weight=sample_weights_tst, average='weighted')
                best_gs = grids[idx]
                best_clf = idx
        print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])

        # Save best grid search pipeline to file
        dump_file = './models_w_newd/' + label + '_best_gs_pipeline.pkl'
        with open(dump_file, 'wb') as ofile:
            pickle.dump(best_gs, ofile)
        print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
