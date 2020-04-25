import numpy as np
import os
import pandas as pd
from expermientos1 import prepare_data, fillna, to_numeric
from sklearn.metrics import classification_report
import pickle

def main():
    data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')
    data_m = pd.read_csv('/home/jose/Escritorio/datathon/src/data/transformed_input.csv', index_col='ID')

    labels = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)
    is_test = data_m['is_test']

    data.drop(['Q_R_4_0_0', 'Q_R_4_0_1', 'Q_R_4_0_2', 'Q_R_4_0_3', 'Q_R_4_0_4',
       'Q_R_4_0_5', 'Q_R_4_0_6', 'Q_R_4_0_7', 'Q_R_4_0_8', 'Q_R_4_0_9',
       'Q_R_4_1_0', 'Q_G_3_0_0', 'Q_G_3_0_1', 'Q_G_3_0_2', 'Q_G_3_0_3',
       'Q_G_3_0_4', 'Q_G_3_0_5', 'Q_G_3_0_6', 'Q_G_3_0_7', 'Q_G_3_0_8',
       'Q_G_3_0_9', 'Q_G_3_1_0', 'Q_B_2_0_0', 'Q_B_2_0_1', 'Q_B_2_0_2',
       'Q_B_2_0_3', 'Q_B_2_0_4', 'Q_B_2_0_5', 'Q_B_2_0_6', 'Q_B_2_0_7',
       'Q_B_2_0_8', 'Q_B_2_0_9', 'Q_B_2_1_0', 'Q_NIR_8_0_0', 'Q_NIR_8_0_1',
       'Q_NIR_8_0_2', 'Q_NIR_8_0_3', 'Q_NIR_8_0_4', 'Q_NIR_8_0_5',
       'Q_NIR_8_0_6', 'Q_NIR_8_0_7', 'Q_NIR_8_0_8', 'Q_NIR_8_0_9',
       'Q_NIR_8_1_0','GEOM_R1', 'GEOM_R2', 'GEOM_R3','GEOM_R4'], axis=1, inplace=True)
    data_m.drop('is_test', axis=1, inplace=True)
    data = pd.concat([data, data_m], axis=1)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    X_train, y_train = data[is_test==False], labels[is_test==False]
    X_test, y_test = data[is_test == True], labels[is_test == True]

    labels_names = np.unique(labels)

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
            y_train_bin = np.array([1 if x == label else -1 for x in y_train])
        else:
            y_train_bin = np.array([-1 if x == label else 1 for x in y_train])

        print('Training...')
        model.fit(X_train, y_train_bin)

        pred_proba = model.predict_proba(X_test)

        if label != 'RESIDENTIAL':
            y_pred_label.append(pred_proba[:, 1])
        else:
            y_pred_label.append(pred_proba[:, 0])

    y_pred = labels_names[np.argmax(y_pred_label, axis=0)]

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()