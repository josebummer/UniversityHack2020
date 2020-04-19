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


def main():
    data = pd.read_csv('data/new_train.txt', sep='|', index_col='ID')
    sample_weights = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    labels = data['CLASE']
    data.drop('CLASE', axis=1, inplace=True)

    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(data))

    y_pred = np.empty(labels.shape, dtype=np.chararray)
    for i, (idx_train, idx_test) in enumerate(folds):
        print('\nFold %d:' % i)

        model = Pipeline([('scl', StandardScaler()),
                          ('clf', RandomForestClassifier(random_state=55, n_jobs=-1))])

        print('Training...')
        model.fit(data.iloc[idx_train], labels[idx_train])

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