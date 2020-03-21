import os
import pickle

import numpy as np
import pandas as pd
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def main():
    data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')
    labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    # data, test, labels_ini, y_test = train_test_split(data, labels_ini, test_size=0.9, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    labels_names = np.unique(labels_ini)

    print('---------------------------------OVA MODEL--------------------------------')
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

            print('Training...')
            model.fit(data.iloc[idx_train], labels[idx_train])

            print('Predicting...')
            pred_proba = model.predict_proba(data.iloc[idx_test])
            pred = model.predict(data.iloc[idx_test])

            print('Local clasification report:')
            print(classification_report(labels[idx_test], pred))

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])

        y_pred[idx_test] = np.argmax(y_pred_label, axis=0)

        print('Fold classification report:')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]]))

    assert -1 not in np.unique(y_pred)

    print('Global classification report:')
    print(classification_report(labels_ini, labels_names[y_pred]))

    with open('predictions/OVA.pkl', 'wb') as ofile:
        pickle.dump(labels_names[y_pred], ofile)

    print('---------------------------------KNN MODEL--------------------------------')

    sc = StandardScaler()
    data = pd.DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)

    y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    for k in [1, 3]:
        print('-----K value = %d------' % k)
        for i, (idx_train, idx_test) in enumerate(folds):
            print('\nFold %d:' % i)

            best_params = {'weights': 'distance', 'n_neighbors': k, 'n_jobs': -1, 'metric': 'manhattan'}
            model = KNeighborsClassifier(**best_params)

            print('Training...')
            model.fit(data.iloc[idx_train], labels_ini[idx_train])

            print('Predicting...')
            pred = model.predict(data.iloc[idx_test])

            y_pred[idx_test] = pred

            print('Fold classification report:')
            print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))

        assert 0 not in np.unique(y_pred)

        print('Global classification report:')
        print(classification_report(labels_ini, y_pred))

        with open('predictions/KNN.pkl', 'wb') as ofile:
            pickle.dump(y_pred, ofile)

    print('---------------------------------Predict OVA MODEL--------------------------------')

    data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/trainOVA.txt', sep='|', index_col='ID')
    labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    sc = StandardScaler()
    data = pd.DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)

    y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    best_params = {'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.01, 'n_jobs': -1}

    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        model = Pipeline([('scl', StandardScaler()),
                          ('enn', EditedNearestNeighbours(sampling_strategy='majority', n_jobs=-1)),
                          ('clf', XGBClassifier(**best_params))])

        print('Training...')
        model.fit(data.iloc[idx_train], labels_ini[idx_train])

        print('Predicting...')
        pred = model.predict(data.iloc[idx_test])

        y_pred[idx_test] = pred

        print('Fold classification report:')
        print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))

    assert 0 not in np.unique(y_pred)

    print('Global classification report:')
    print(classification_report(labels_ini, y_pred))

    print('---------------------------------Raw + Predict OVA MODEL--------------------------------')

    data_raw = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')
    data_raw.drop('CLASE', axis=1, inplace=True)
    data = pd.concat([data_raw, data], axis=1)

    sc = StandardScaler()
    data = pd.DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)

    y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    best_params = {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01, 'n_jobs': -1}

    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        model = Pipeline([('scl', StandardScaler()),
                          ('enn', EditedNearestNeighbours(sampling_strategy='majority', n_jobs=-1)),
                          ('clf', XGBClassifier(**best_params))])

        print('Training...')
        model.fit(data.iloc[idx_train], labels_ini[idx_train])

        print('Predicting...')
        pred = model.predict(data.iloc[idx_test])

        y_pred[idx_test] = pred

        print('Fold classification report:')
        print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))

    assert 0 not in np.unique(y_pred)

    print('Global classification report:')
    print(classification_report(labels_ini, y_pred))


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
