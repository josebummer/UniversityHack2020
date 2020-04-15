import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
import progressbar
from expermientos1 import prepare_data, fillna, to_numeric
import os


def main():
    nb = ComplementNB()
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')

    labels = train['CLASE']
    train.drop('CLASE', axis=1, inplace=True)

    train = prepare_data(train)
    train = fillna(train)
    train = to_numeric(train)

    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)

    pipe_gb = Pipeline([('scl', StandardScaler()),
                        ('clf', GaussianNB())])

    pipe_mnb = Pipeline([('scl', MinMaxScaler()),
                         ('clf', MultinomialNB())])

    pipe_cnb = Pipeline([('scl', MinMaxScaler()),
                         ('clf', ComplementNB())])

    pipe_bnb = Pipeline([('scl', StandardScaler()),
                         ('clf', BernoulliNB())])

    grid_params_gb = [{'clf__var_smoothing': np.linspace(1e-11,1e-6)}]

    grid_params_mnb = [{'clf__alpha': np.linspace(0,4),
                        'clf__fit_prior': [True, False]}]

    grid_params_cnb = [{'clf__alpha': np.linspace(0,4),
                        'clf__fit_prior': [True, False],
                        'clf__norm': [True, False]}]

    grid_params_bnb = [{'clf__alpha': np.linspace(0,4),
                        'clf__fit_prior': [True, False]}]

    jobs = -1

    gs_gb = GridSearchCV(estimator=pipe_gb,
                           param_grid=grid_params_gb,
                           scoring=make_scorer(f1_score, average='macro'),
                           cv=5,
                           n_jobs=jobs,)

    gs_cnb = GridSearchCV(estimator=pipe_cnb,
                             param_grid=grid_params_cnb,
                             scoring=make_scorer(f1_score, average='macro'),
                             cv=5,
                             n_jobs=jobs)

    gs_mnb = GridSearchCV(estimator=pipe_mnb,
                            param_grid=grid_params_mnb,
                            scoring=make_scorer(f1_score, average='macro'),
                            cv=5,
                            n_jobs=jobs)

    gs_bnb = GridSearchCV(estimator=pipe_bnb,
                          param_grid=grid_params_bnb,
                          scoring=make_scorer(f1_score, average='macro'),
                          cv=5,
                          n_jobs=jobs)

    grids = [gs_gb, gs_cnb, gs_mnb, gs_bnb]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Gaussian Naive Bayes', 1: 'Complement Naive Bayes',
                 2: 'Multinomial Naive Bayes', 3: 'Bernuilli Naive Bayes'}

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
        # print('Weighted clasification:')
        # print(classification_report(y_test, y_pred, sample_weight=sample_weights_tst))
        # Track best (highest test f1) model
        if f1_score(y_test, y_pred, average='macro') > best_f1:
            best_f1 = f1_score(y_test, y_pred, average='macro')
            best_gs = grids[idx]
            best_clf = idx
    print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])

if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()