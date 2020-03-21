import os
import pickle

import numpy as np
import pandas as pd
import progressbar
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main():
    data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')

    labels = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    train = prepare_data(data)
    train = fillna(train)
    train = to_numeric(train)

    labels_names = np.unique(labels)

    CREATE_DATA = False

    if CREATE_DATA:
        y_pred_label = []
        for label in progressbar.progressbar(labels_names):
            print('Load %s model:' % label)

            dump_file = './models/' + label + '_best_gs_pipeline.pkl'
            with open(dump_file, 'rb') as ofile:
                grid = pickle.load(ofile)

            model = grid.best_estimator_
            for step in model.steps:
                if step[0] in ['enn', 'clf']:
                    step[1].n_jobs = -1

            # if label != 'RESIDENTIAL':
            y_train = np.array([1 if x == label else -1 for x in labels])
            # else:
            #     y_train = np.array([-1 if x == label else 1 for x in labels])

            print('Predicting...')
            pred_proba = cross_val_predict(model, train, y_train, method='predict_proba', n_jobs=-1, cv=5)

            # if label != 'RESIDENTIAL':
            y_pred_label.append(pred_proba[:, 1])
            # else:
            #     y_pred_label.append(pred_proba[:, 0])

        n_train = pd.DataFrame(np.array(y_pred_label).T, index=train.index, columns=labels_names)

        print('Save new train')
        pd.concat([n_train, labels], axis=1).to_csv('data/trainOVA.txt', sep='|')
    else:
        print('Load new data')
        n_train = pd.read_csv('/home/jose/Escritorio/datathon/src/data/trainOVA.txt', sep='|', index_col='ID')

        labels = n_train.iloc[:, -1]
        n_train.drop('CLASE', axis=1, inplace=True)

    ####################################################################################################################

    X_train, X_test, y_train, y_test = train_test_split(n_train, labels, test_size=0.2, random_state=42)

    pipe_xgb = Pipeline([('scl', StandardScaler()),
                         ('clf', XGBClassifier(random_state=42))])

    pipe_xgb_noisy = Pipeline([('scl', StandardScaler()),
                               ('enn', EditedNearestNeighbours(random_state=42, sampling_strategy='majority')),
                               ('clf', XGBClassifier(random_state=42))])

    pipe_xgb_pca = Pipeline([('scl', StandardScaler()),
                             ('pca', PCA(n_components=3)),
                             ('clf', XGBClassifier(random_state=42))])

    grid_params_xgboost = [{'clf__max_depth': [2, 6, 12, 18],
                            'clf__learning_rate': [0.01, 0.1, 0.15],
                            'clf__n_estimators': [400, 600, 1000, 1200]}]

    jobs = -1
    gs_xgb = RandomizedSearchCV(estimator=pipe_xgb,
                                param_distributions=grid_params_xgboost,
                                scoring=make_scorer(f1_score, average='macro'),
                                cv=5,
                                n_jobs=jobs,
                                n_iter=20)

    gs_xgb_noisy = RandomizedSearchCV(estimator=pipe_xgb_noisy,
                                      param_distributions=grid_params_xgboost,
                                      scoring=make_scorer(f1_score, average='macro'),
                                      cv=5,
                                      n_jobs=jobs,
                                      n_iter=20)

    gs_xgb_pca = RandomizedSearchCV(estimator=pipe_xgb_pca,
                                    param_distributions=grid_params_xgboost,
                                    scoring=make_scorer(f1_score, average='macro'),
                                    cv=5,
                                    n_jobs=jobs,
                                    n_iter=20)

    grids = [gs_xgb, gs_xgb_noisy, gs_xgb_pca]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'XGB', 1: 'XGB w/ENN', 2: 'XGB w/PCA'}

    # Fit the grid search objects
    print('-------------------------------------------Only predict OVA------------------------------------------------')
    print('Performing model optimizations...')
    best_f1 = 0.0
    best_clf = 0
    for idx in progressbar.progressbar(range(len(grids))):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        grids[idx].fit(X_train, y_train)
        # Best params
        print('Best params: %s' % grids[idx].best_params_)
        # Best training data accuracy
        print('Best training f1: %.3f' % grids[idx].best_score_)
        # Predict on test data with best params
        model = grid.best_estimator_
        for step in model.steps:
            if step[0] in ['enn', 'clf']:
                step[1].n_jobs = -1
        y_pred = model.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set metrics for best params:')
        print(classification_report(y_test, y_pred))
        # Track best (highest test f1) model
        if f1_score(y_test, y_pred, average='macro') > best_f1:
            best_f1 = f1_score(y_test, y_pred, average='macro')
            best_clf = idx
    print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])

    ########################################################################################################################

    n_data = pd.concat([train, n_train], axis=1)
    X_train = n_data.loc[X_train.index]
    X_test = n_data.loc[X_test.index]

    print('--------------------------------------------OVA + Raw data-------------------------------------------------')
    print('Performing model optimizations...')
    best_f1 = 0.0
    best_clf = 0
    for idx in progressbar.progressbar(range(len(grids))):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        grids[idx].fit(X_train, y_train)
        # Best params
        print('Best params: %s' % grids[idx].best_params_)
        # Best training data accuracy
        print('Best training f1: %.3f' % grids[idx].best_score_)
        # Predict on test data with best params
        model = grid.best_estimator_
        for step in model.steps:
            if step[0] in ['enn', 'clf']:
                step[1].n_jobs = -1
        y_pred = model.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set metrics for best params:')
        print(classification_report(y_test, y_pred))
        # Track best (highest test f1) model
        if f1_score(y_test, y_pred, average='macro') > best_f1:
            best_f1 = f1_score(y_test, y_pred, average='macro')
            best_clf = idx
    print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
