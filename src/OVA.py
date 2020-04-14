import os
import pickle
import json

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
    data = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    # data, test, labels_ini, y_test = train_test_split(data, labels_ini, test_size=0.9, random_state=42)

    #todos semilla = 42
    kf = KFold(n_splits=5, shuffle=True, random_state=58)
    folds = list(kf.split(data))

    labels_names = np.unique(labels_ini)

    # print('---------------------------------OVA MODEL--------------------------------')
    # y_pred = np.ones(labels_ini.shape[0], dtype=np.int) * -1
    # for i, (idx_train, idx_test) in enumerate(folds):
    #     print('\nFold %d:' % i)
    #
    #     y_pred_label = []
    #     for label in labels_names:
    #         print('Load %s model:' % label)
    #
    #         dump_file = './models/' + label + '_best_gs_pipeline.pkl'
    #         with open(dump_file, 'rb') as ofile:
    #             grid = pickle.load(ofile)
    #
    #         model = grid.best_estimator_
    #         for step in model.steps:
    #             if step[0] in ['enn', 'clf']:
    #                 step[1].n_jobs = -1
    #
    #         if label != 'RESIDENTIAL':
    #             labels = np.array([1 if x == label else -1 for x in labels_ini])
    #         else:
    #             labels = np.array([-1 if x == label else 1 for x in labels_ini])
    #
    #         print('Training...')
    #         model.fit(data.iloc[idx_train], labels[idx_train])
    #
    #         print('Predicting...')
    #         pred_proba = model.predict_proba(data.iloc[idx_test])
    #         pred = model.predict(data.iloc[idx_test])
    #
    #         print('Local clasification report:')
    #         print(classification_report(labels[idx_test], pred))
    #
    #         if label != 'RESIDENTIAL':
    #             y_pred_label.append(pred_proba[:, 1])
    #         else:
    #             y_pred_label.append(pred_proba[:, 0])
    #
    #     y_pred[idx_test] = np.argmax(y_pred_label, axis=0)
    #
    #     print('Fold classification report:')
    #     print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]]))
    #
    # assert -1 not in np.unique(y_pred)
    #
    # print('Global classification report:')
    # print(classification_report(labels_ini, labels_names[y_pred]))
    #
    # with open('predictions/OVA.pkl', 'wb') as ofile:
    #     pickle.dump(labels_names[y_pred], ofile)
    #
    # print('---------------------------------KNN MODEL--------------------------------')
    #
    # sc = StandardScaler()
    # data = pd.DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)
    #
    # y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    # for k in [1, 3]:
    #     print('-----K value = %d------' % k)
    #     for i, (idx_train, idx_test) in enumerate(folds):
    #         print('\nFold %d:' % i)
    #
    #         best_params = {'weights': 'distance', 'n_neighbors': k, 'n_jobs': -1, 'metric': 'manhattan'}
    #         model = KNeighborsClassifier(**best_params)
    #
    #         print('Training...')
    #         model.fit(data.iloc[idx_train], labels_ini[idx_train])
    #
    #         print('Predicting...')
    #         pred = model.predict(data.iloc[idx_test])
    #
    #         y_pred[idx_test] = pred
    #
    #         print('Fold classification report:')
    #         print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))
    #
    #     assert 0 not in np.unique(y_pred)
    #
    #     print('Global classification report:')
    #     print(classification_report(labels_ini, y_pred))
    #
    #     with open('predictions/KNN.pkl', 'wb') as ofile:
    #         pickle.dump(y_pred, ofile)

    # print('---------------------------------Predict OVA MODEL--------------------------------')
    #
    # data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/trainOVA.txt', sep='|', index_col='ID')
    # labels_ini = data.iloc[:, -1]
    # data.drop('CLASE', axis=1, inplace=True)
    #
    # y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    # best_params = {'n_estimators': 600, 'max_depth': 2, 'learning_rate': 0.1, 'n_jobs': -1}
    #
    # for i, (idx_train, idx_test) in enumerate(folds):
    #     print('\nFold %d:' % i)
    #
    #     model = Pipeline([('scl', StandardScaler()),
    #                       ('enn', EditedNearestNeighbours(sampling_strategy='majority', n_jobs=-1)),
    #                       ('clf', XGBClassifier(**best_params))])
    #
    #     print('Training...')
    #     model.fit(data.iloc[idx_train], labels_ini[idx_train])
    #
    #     print('Predicting...')
    #     pred = model.predict(data.iloc[idx_test])
    #
    #     y_pred[idx_test] = pred
    #
    #     print('Fold classification report:')
    #     print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))
    #
    # assert 0 not in np.unique(y_pred)
    #
    # print('Global classification report:')
    # print(classification_report(labels_ini, y_pred))
    #
    # print('---------------------------------Raw + Predict OVA MODEL--------------------------------')
    #
    # data_raw = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')
    # data_raw.drop('CLASE', axis=1, inplace=True)
    #
    # data_raw = prepare_data(data_raw)
    # data_raw = fillna(data_raw)
    # data_raw = to_numeric(data_raw)
    #
    # data = pd.concat([data_raw, data], axis=1)
    #
    # y_pred = np.zeros(labels_ini.shape[0], dtype=np.chararray)
    # best_params = {'n_estimators': 600, 'max_depth': 12, 'learning_rate': 0.4, 'n_jobs': -1}
    #
    # for i, (idx_train, idx_test) in enumerate(folds):
    #     print('\nFold %d:' % i)
    #
    #     model = Pipeline([('scl', StandardScaler()),
    #                       ('enn', EditedNearestNeighbours(sampling_strategy='majority', n_jobs=-1)),
    #                       ('clf', XGBClassifier(**best_params))])
    #
    #     print('Training...')
    #     model.fit(data.iloc[idx_train], labels_ini[idx_train])
    #
    #     print('Predicting...')
    #     pred = model.predict(data.iloc[idx_test])
    #
    #     y_pred[idx_test] = pred
    #
    #     print('Fold classification report:')
    #     print(classification_report(labels_ini.iloc[idx_test], y_pred[idx_test]))
    #
    # assert 0 not in np.unique(y_pred)
    #
    # print('Global classification report:')
    # print(classification_report(labels_ini, y_pred))

    # print('---------------------------------OVA-Weighted MODEL--------------------------------')
    # data = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    # sample_weight = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    # labels_ini = data['CLASE']
    # data.drop('CLASE', axis=1, inplace=True)
    #
    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)
    #
    # class_weights_factor = {'RESIDENTIAL': 4.812552140340716e-06*data.shape[0],
    #                  'INDUSTRIAL': 4.647398736012043e-05*data.shape[0],
    #                  'PUBLIC': 3.783937948148589e-05*data.shape[0],
    #                  'OFFICE': 4.558736182249404e-05*data.shape[0],
    #                  'RETAIL': 4.2627096025849134e-05*data.shape[0],
    #                  'AGRICULTURE': 6.261938403426534e-05*data.shape[0],
    #                  'OTHER': 3.8319803354362536e-05*data.shape[0]}
    #
    # class_weights = {'RESIDENTIAL': 4.812552140340716e-06,
    #                  'INDUSTRIAL': 4.647398736012043e-05,
    #                  'PUBLIC': 3.783937948148589e-05,
    #                  'OFFICE': 4.558736182249404e-05,
    #                  'RETAIL': 4.2627096025849134e-05,
    #                  'AGRICULTURE': 6.261938403426534e-05,
    #                  'OTHER': 3.8319803354362536e-05}
    #
    # sample_weight_factor = np.array([class_weights_factor[i] for i in labels_ini])
    #
    # y_pred = np.ones(labels_ini.shape[0], dtype=np.int) * -1
    # for i, (idx_train, idx_test) in enumerate(folds):
    #     print('\nFold %d:' % i)
    #
    #     y_pred_label = []
    #     for label in labels_names:
    #         print('Load %s model:' % label)
    #
    #         dump_file = './models_nw_dg/' + label + '_best_gs_pipeline.pkl'
    #         with open(dump_file, 'rb') as ofile:
    #             grid = pickle.load(ofile)
    #
    #         model = grid.best_estimator_
    #         for step in model.steps:
    #             if step[0] in ['enn', 'clf']:
    #                 step[1].n_jobs = -1
    #
    #         if label != 'RESIDENTIAL':
    #             labels = np.array([1 if x == label else -1 for x in labels_ini])
    #         else:
    #             labels = np.array([-1 if x == label else 1 for x in labels_ini])
    #
    #         class_weight = {}
    #         if label == 'RESIDENTIAL':
    #             class_weight[-1] = class_weights['RESIDENTIAL']
    #             class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
    #         else:
    #             class_weight[1] = class_weights[label]
    #             class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
    #
    #         class_weight_factor = {}
    #         if label == 'RESIDENTIAL':
    #             class_weight_factor[-1] = class_weights_factor['RESIDENTIAL']
    #             class_weight_factor[1] = np.sum([class_weights_factor[value] for value in class_weights_factor.keys() if value != label])
    #         else:
    #             class_weight_factor[1] = class_weights_factor[label]
    #             class_weight_factor[-1] = np.sum([class_weights_factor[value] for value in class_weights_factor.keys() if value != label])
    #
    #         sample_weights_bin = np.array([class_weight[i] for i in labels[idx_test]])
    #         sample_weights_bin_factor = np.array([class_weight_factor[i] for i in labels[idx_test]])
    #
    #         print('Training...')
    #         model.fit(data.iloc[idx_train], labels[idx_train])
    #
    #         print('Predicting...')
    #         pred_proba = model.predict_proba(data.iloc[idx_test])
    #         pred = model.predict(data.iloc[idx_test])
    #
    #         print('Local clasification report:')
    #         print('Normal:')
    #         print(classification_report(labels[idx_test], pred))
    #         print('Weigthed:')
    #         print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin))
    #         print('Weigthed factor:')
    #         print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin_factor))
    #
    #         if label != 'RESIDENTIAL':
    #             y_pred_label.append(pred_proba[:, 1])
    #         else:
    #             y_pred_label.append(pred_proba[:, 0])
    #
    #     y_pred[idx_test] = np.argmax(y_pred_label, axis=0)
    #
    #     print('Fold classification report:')
    #     print('Normal')
    #     print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]]))
    #     print('Weigthed:')
    #     print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]], sample_weight=sample_weight.iloc[idx_test]))
    #     print('Weigthed factor:')
    #     print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]],
    #                                 sample_weight=sample_weight_factor[idx_test]))
    #
    # assert -1 not in np.unique(y_pred)
    #
    # print('Global classification report:')
    # print('Normal:')
    # print(classification_report(labels_ini, labels_names[y_pred]))
    # print('Weigthed:')
    # print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight))
    # print('Weigthed factor:')
    # print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight_factor))

    print('---------------------------------OVA-Weighted-Genetic MODEL--------------------------------')
    data = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    sample_weight = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    labels_ini = data['CLASE']
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    class_weights_factor = {'RESIDENTIAL': 4.812552140340716e-06 * data.shape[0],
                            'INDUSTRIAL': 4.647398736012043e-05 * data.shape[0],
                            'PUBLIC': 3.783937948148589e-05 * data.shape[0],
                            'OFFICE': 4.558736182249404e-05 * data.shape[0],
                            'RETAIL': 4.2627096025849134e-05 * data.shape[0],
                            'AGRICULTURE': 6.261938403426534e-05 * data.shape[0],
                            'OTHER': 3.8319803354362536e-05 * data.shape[0]}

    class_weights = {'RESIDENTIAL': 4.812552140340716e-06,
                     'INDUSTRIAL': 4.647398736012043e-05,
                     'PUBLIC': 3.783937948148589e-05,
                     'OFFICE': 4.558736182249404e-05,
                     'RETAIL': 4.2627096025849134e-05,
                     'AGRICULTURE': 6.261938403426534e-05,
                     'OTHER': 3.8319803354362536e-05}

    sample_weight_factor = np.array([class_weights_factor[i] for i in labels_ini])

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

            dump_file = './models_nw_dg/' + label + '_best_gs_pipeline.pkl'
            with open(dump_file, 'rb') as ofile:
                grid = pickle.load(ofile)

            model = grid.best_estimator_
            for step in model.steps:
                if step[0] in ['enn', 'clf']:
                    step[1].n_jobs = -1

            dump_file = 'GEN_OUTPUT_' + label + '.json'
            with open(dump_file, 'r') as ofile:
                mask = json.load(ofile)

            best_cols = np.array(mask['best_x'][np.argmin(mask['best_f'])]).astype(np.bool)

            if label == 'AGRICULTURE':
                best_cols[1] = True

            data_mask = data.loc[:, best_cols]

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

            class_weight_factor = {}
            if label == 'RESIDENTIAL':
                class_weight_factor[-1] = class_weights_factor['RESIDENTIAL']
                class_weight_factor[1] = np.sum(
                    [class_weights_factor[value] for value in class_weights_factor.keys() if value != label])
            else:
                class_weight_factor[1] = class_weights_factor[label]
                class_weight_factor[-1] = np.sum(
                    [class_weights_factor[value] for value in class_weights_factor.keys() if value != label])

            sample_weights_bin = np.array([class_weight[i] for i in labels[idx_test]])
            sample_weights_bin_factor = np.array([class_weight_factor[i] for i in labels[idx_test]])

            print('Training...')
            model.fit(data_mask.iloc[idx_train], labels[idx_train])

            print('Predicting...')
            pred_proba = model.predict_proba(data_mask.iloc[idx_test])
            pred = model.predict(data_mask.iloc[idx_test])

            print('Local clasification report:')
            print('Normal:')
            print(classification_report(labels[idx_test], pred, digits=3))
            print('Weigthed:')
            print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin, digits=3))
            print('Weigthed factor:')
            print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin_factor, digits=3))

            if label != 'RESIDENTIAL':
                y_pred_label.append(pred_proba[:, 1])
            else:
                y_pred_label.append(pred_proba[:, 0])

            y_pred_label_bin[label][idx_test] = pred

        y_pred[idx_test] = np.argmax(y_pred_label, axis=0)

        print('Fold classification report:')
        print('Normal')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]], digits=3))
        print('Weigthed:')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]],
                                    sample_weight=sample_weight.iloc[idx_test], digits=3))
        print('Weigthed factor:')
        print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]],
                                    sample_weight=sample_weight_factor[idx_test], digits=3))

    assert -1 not in np.unique(y_pred)

    for label in labels_names:
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

        class_weight_factor = {}
        if label == 'RESIDENTIAL':
            class_weight_factor[-1] = class_weights_factor['RESIDENTIAL']
            class_weight_factor[1] = np.sum(
                [class_weights_factor[value] for value in class_weights_factor.keys() if value != label])
        else:
            class_weight_factor[1] = class_weights_factor[label]
            class_weight_factor[-1] = np.sum(
                [class_weights_factor[value] for value in class_weights_factor.keys() if value != label])

        sample_weights_bin = np.array([class_weight[i] for i in labels])
        sample_weights_bin_factor = np.array([class_weight_factor[i] for i in labels])

        assert -2 not in np.unique(y_pred_label_bin[label])

        print('Label binary ' + label + ' classification report:')
        print('Normal:')
        print(classification_report(labels, y_pred_label_bin[label], digits=3))
        print('Weigthed:')
        print(classification_report(labels, y_pred_label_bin[label], sample_weight=sample_weights_bin, digits=3))
        print('Weigthed factor:')
        print(classification_report(labels, y_pred_label_bin[label], sample_weight=sample_weights_bin_factor, digits=3))

    print('Global classification report:')
    print('Normal:')
    print(classification_report(labels_ini, labels_names[y_pred], digits=3))
    print('Weigthed:')
    print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight, digits=3))
    print('Weigthed factor:')
    print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight_factor, digits=3))


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()
