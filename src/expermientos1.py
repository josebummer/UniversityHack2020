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

from aux_scorer import get_weight_f1

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
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    labels_ini = train.iloc[:, -1]
    train.drop('CLASE', axis=1, inplace=True)

    train = prepare_data(train)
    train = fillna(train)
    train = to_numeric(train)

    weights = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    class_weights = {'RESIDENTIAL': 4.812552140340716e-06*train.shape[0],
                    'INDUSTRIAL': 4.647398736012043e-05*train.shape[0],
                    'PUBLIC': 3.783937948148589e-05*train.shape[0],
                    'OFFICE': 4.558736182249404e-05*train.shape[0],
                    'RETAIL': 4.2627096025849134e-05*train.shape[0],
                    'AGRICULTURE': 6.261938403426534e-05*train.shape[0],
                    'OTHER': 3.8319803354362536e-05*train.shape[0]}

    # train, test, yy_train, yy_test = train_test_split(train, labels_ini, test_size=0.2, random_state=42)

    for label in progressbar.progressbar(np.unique(labels_ini)):
        print('\n--------------------OVA: %s vs All------------------' % label)
        # if label != 'RESIDENTIAL':
        #     labels = np.array([1 if x == label else -1 for x in labels_ini])
        # else:
        #     labels = np.array([-1 if x == label else 1 for x in labels_ini])
        labels = np.array([1 if x == label else -1 for x in labels_ini])

        class_weight = {}
        # if label == 'RESIDENTIAL':
        #     class_weight[-1] = class_weights['RESIDENTIAL']
        #     class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
        # else:
        #     class_weight[1] = class_weights[label]
        #     class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
        class_weight[1] = class_weights[label]
        class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])

        # f1w_scorer = get_weight_f1(class_weight)
        f1w = get_weight_f1(class_weight)
        f1w_scorer = make_scorer(f1w)

        X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)
        # sample_weights_tr = np.array([class_weight[i] for i in y_train])
        sample_weights_tst = np.array([class_weight[i] for i in y_test])

        # Construct some pipelines
        pipe_rf = Pipeline([('scl', StandardScaler()),
                            ('clf', RandomForestClassifier(random_state=42, class_weight=class_weight))])

        pipe_rf_noisy = Pipeline([('scl', StandardScaler()),
                                  ('enn', EditedNearestNeighbours(random_state=42, sampling_strategy='majority')),
                                  ('clf', RandomForestClassifier(random_state=42, class_weight=class_weight))])

        pipe_knn = Pipeline([('scl', StandardScaler()),
                             ('clf', KNeighborsClassifier())])

        pipe_knn_noisy = Pipeline([('scl', StandardScaler()),
                                   ('enn', EditedNearestNeighbours(random_state=42, sampling_strategy='majority')),
                                   ('clf', KNeighborsClassifier())])

        pipe_xgb = Pipeline([('scl', StandardScaler()),
                             ('clf', XGBClassifier(random_state=42, class_weight=class_weight))])

        pipe_xgb_noisy = Pipeline([('scl', StandardScaler()),
                                   ('enn', EditedNearestNeighbours(random_state=42, sampling_strategy='majority')),
                                   ('clf', XGBClassifier(random_state=42, class_weight=class_weight))])

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

        gs_rf_noisy = RandomizedSearchCV(estimator=pipe_rf_noisy,
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

        gs_knn_noisy = RandomizedSearchCV(estimator=pipe_knn_noisy,
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

        gs_xgb_noisy = RandomizedSearchCV(estimator=pipe_xgb_noisy,
                                          param_distributions=grid_params_xgboost,
                                          scoring=f1w_scorer,
                                          cv=5,
                                          n_jobs=jobs,
                                          n_iter=20)

        # List of pipelines for ease of iteration
        grids = [gs_rf, gs_rf_noisy, gs_knn, gs_knn_noisy, gs_xgb, gs_xgb_noisy]

        # Dictionary of pipelines and classifier types for ease of reference
        grid_dict = {0: 'Random Forest', 1: 'Random Forest w/ENN',
                     2: 'KNN', 3: 'KNN w/ENN',
                     4: 'XGB', 5: 'XGB w/ENN'}

        # Fit the grid search objects
        print('Performing model optimizations...')
        best_f1 = 0.0
        best_clf = 0
        best_gs = ''
        for idx in progressbar.progressbar(range(len(grids))):
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
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
            if f1_score(y_test, y_pred, sample_weight=sample_weights_tst) > best_f1:
                best_f1 = f1_score(y_test, y_pred, sample_weight=sample_weights_tst)
                best_gs = grids[idx]
                best_clf = idx
        print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])

        # Save best grid search pipeline to file
        dump_file = './models_w_dg_factor/' + label + '_best_gs_pipeline.pkl'
        with open(dump_file, 'wb') as ofile:
            pickle.dump(best_gs, ofile)
        print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
