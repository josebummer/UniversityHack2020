import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from expermientos1 import prepare_data, fillna, to_numeric
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from aux_scorer import get_weight_f1


def main():
    data = pd.read_csv('data/df_final.csv', sep='|', index_col='ID')
    sample_weights = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')
    data = data.iloc[:sample_weights.shape[0], ]
    labels = data['CLASE']
    data.drop('CLASE', axis=1, inplace=True)

    basic_cols = ['CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID', 'AREA']
    rgbn_cols = [x for x in data.columns if "RGBN_PROB_" in x]
    geom_ori_cols = [x for x in data.columns if "GEOM_R" in x]
    geom_dist_cols = [x for x in data.columns if "GEOM_DIST4" in x]
    geom_prob_cols = [x for x in data.columns if "GEOM_PROB" in x]
    xy_dens_cols = [x for x in data.columns if "XY_DENS" in x]
    xy_ori_cols = ['X', 'Y']

    custom_cols = [f'CUSTOM_{i}' for i in range(4)]
    distances_cols = [f'C_DIST_{i}' for i in range(21)]

    mod_cols = basic_cols + geom_ori_cols + geom_dist_cols + geom_prob_cols + rgbn_cols + xy_dens_cols + custom_cols + distances_cols + xy_ori_cols

    data = data[mod_cols]

    # data = prepare_data(data)
    # data = fillna(data)
    # data = to_numeric(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    class_weight = {'RESIDENTIAL': 4.812552140340716e-06 * labels.size,
                     'INDUSTRIAL': 4.647398736012043e-05 * labels.size,
                     'PUBLIC': 3.783937948148589e-05 * labels.size,
                     'OFFICE': 4.558736182249404e-05 * labels.size,
                     'RETAIL': 4.2627096025849134e-05 * labels.size,
                     'AGRICULTURE': 6.261938403426534e-05 * labels.size,
                     'OTHER': 3.8319803354362536e-05 * labels.size}

    sample_weights_tr = np.array([class_weight[i] for i in y_train])
    sample_weights_tst = np.array([class_weight[i] for i in y_test])

    f1w = get_weight_f1(class_weight)
    f1w_scorer = make_scorer(f1w)

    grid_params_xgboost = [{
        'learning_rate': [0.05, 0.1, 0.15],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 6, 10],
        'n_estimators': [100, 200, 400, 600]
    }]

    gs_xgb = RandomizedSearchCV(estimator=XGBClassifier(tree_method='hist'),
                            param_distributions=grid_params_xgboost,
                            scoring=f1w_scorer,
                            cv=5,
                            n_jobs=16,
                            n_iter=40)

    gs_xgb.fit(X_train, y_train, sample_weight=sample_weights_tr)
    # Best params
    print('Best params: %s' % gs_xgb.best_params_)
    # Best training data accuracy
    print('Best training f1: %.3f' % gs_xgb.best_score_)
    # Predict on test data with best params
    y_pred = gs_xgb.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set metrics for best params:')
    print('Normal clasification:')
    print(classification_report(y_test, y_pred))
    print('Weighted clasification:')
    print(classification_report(y_test, y_pred, sample_weight=sample_weights_tst))

if __name__=='__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()