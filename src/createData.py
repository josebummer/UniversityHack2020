import numpy as np
import os
import pandas as pd
import pickle

from expermientos1 import prepare_data, fillna, to_numeric


def best_models_and_labels():
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')

    labels_ini = train.iloc[:, -1]

    labels_names = np.unique(labels_ini)

    models = {}
    labels = {}
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
            y_train = np.array([1 if x == label else -1 for x in labels_ini])
        else:
            y_train = np.array([-1 if x == label else 1 for x in labels_ini])

        models[label] = (model)
        labels[label] = (y_train)

    return models, labels


def main():
    #Si se usa train hay que llamar a los métodos de experimentos1 que se importan arriba.
    #Si se hace con trainGuille no hace falta.

    models, labels = best_models_and_labels()


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()