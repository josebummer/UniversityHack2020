import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import pandas as pd
from expermientos1 import prepare_data, fillna, to_numeric
import pickle


dump_file = 'data/aggregated_mask_gt0.pkl'
with open(dump_file, 'rb') as ofile:
    gt0 = pickle.load(ofile)

gt0['AGRICULTURE'][1] = True
mask_dict = gt0


def main():
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    test = pd.read_csv('data/test.txt', sep='|', index_col='ID')

    labels = train.iloc[:, -1]
    train.drop('CLASE', axis=1, inplace=True)

    data = pd.concat([train, test], sort=False)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    train, test = data.iloc[:train.shape[0], ], data.iloc[train.shape[0]:, ]

    labels_names = np.unique(labels)

    y_pred_label = []
    for label in labels_names:
        print('Load %s model:' % label)

        dump_file = './1models_nw_dn/' + label + '_best_gs_pipeline.pkl'
        with open(dump_file, 'rb') as ofile:
            grid = pickle.load(ofile)

        model = grid.best_estimator_
        for step in model.steps:
            if step[0] in ['enn', 'clf']:
                step[1].n_jobs = -1

        if label != 'RESIDENTIAL':
            y_train = np.array([1 if x == label else -1 for x in labels])
        else:
            y_train = np.array([-1 if x == label else 1 for x in labels])

        print('Training...')
        model.fit(train.loc[:, mask_dict[label]], y_train)

        print('Predicting...')
        pred_proba = model.predict_proba(test.loc[:, mask_dict[label]])

        if label != 'RESIDENTIAL':
            y_pred_label.append(pred_proba[:, 1])
        else:
            y_pred_label.append(pred_proba[:, 0])

    n_data = pd.DataFrame(np.array(y_pred_label).T, index=test.index, columns=labels_names)

    print('Save file')
    n_data.to_csv('data/test_mask_EVERY_MASK.txt', sep='|')
    # pd.concat([n_data, labels], axis=1).to_csv('data/trainOVA-fit.txt', sep='|')

    # y_pred = labels_names[np.argmax(y_pred_label, axis=0)]
    # submit = {'ID': test.index, 'CLASE': y_pred}
    # df_submit = pd.DataFrame(data=submit)
    #
    # df_submit.to_csv('predictions/Minsait_UniversidadGranada_CodeDigger_1.txt', sep='|', index=False)

if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()