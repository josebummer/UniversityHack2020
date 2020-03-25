from sklearn.naive_bayes import GaussianNB
from expermientos1 import prepare_data, fillna, to_numeric
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV, KFold
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report
import pickle

data = pd.read_csv('data/train.txt', sep='|', index_col='ID')

labels_ini = data.iloc[:, -1]
data.drop('CLASE', axis=1, inplace=True)

data = prepare_data(data)
data = fillna(data)
data = to_numeric(data)

# pipe_nb = Pipeline([('scl', StandardScaler()),
#                     ('clf', GaussianNB())])
#
# grid_params_nb = [{'clf__var_smoothing': np.linspace(1e-9, 1e-4)}]
#
# jobs = -1
# gs_rf = GridSearchCV(estimator=pipe_nb,
#                        param_grid=grid_params_nb,
#                        scoring=make_scorer(f1_score, average='macro'),
#                        cv=5,
#                        n_jobs=jobs)
#
# print('\nNaive Bayes')
# # Fit grid search
# gs_rf.fit(data, labels_ini)
# # Best params
# print('Best params: %s' % gs_rf.best_params_)
# # Best training data accuracy
# print('Best training f1: %.3f' % gs_rf.best_score_)
# # Predict on test data with best params
# y_pred = gs_rf.predict(data)
# # Test data accuracy of model with best params
# print('Test set metrics for best params:')
# print(classification_report(labels_ini, y_pred))
# # Track best (highest test f1) model


### (np.where(((OVApredictions.drop('CLASE', axis=1).loc[X_test.index]>0.5).sum(1))>1)[0]).reshape((-1,1))

# X_train, X_test, y_train, y_test = train_test_split(data, labels_ini, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(data))

labels_names = np.unique(labels_ini)

# best_params = {'var_smoothing': 8.367363265306123e-05}
# pipe_nb = Pipeline([('scl', StandardScaler()),
#                     ('clf', GaussianNB(**best_params))])
#
# labels_names = np.unique(labels_ini)
#
# pipe_nb.fit(X_train, y_train)
#
# y_pred = pipe_nb.predict_proba(X_test)
#
# OVAsum = []
# for row in OVApredictions.values:
#     x = row[:-1]
#     OVAsum.append([(np.sum(x*-1) + (2*x[i])) for i in range(len(x))])
#
# print()

sample_weight = pd.read_csv('data/train_weights.cvs', sep='|', index_col='ID')

class_weights = {'RESIDENTIAL': 4.812552140340716e-06*sample_weight.shape[0],
                     'INDUSTRIAL': 4.647398736012043e-05*sample_weight.shape[0],
                     'PUBLIC': 3.783937948148589e-05*sample_weight.shape[0],
                     'OFFICE': 4.558736182249404e-05*sample_weight.shape[0],
                     'RETAIL': 4.2627096025849134e-05*sample_weight.shape[0],
                     'AGRICULTURE': 6.261938403426534e-05*sample_weight.shape[0],
                     'OTHER': 3.8319803354362536e-05*sample_weight.shape[0]}

print('---------------------------------OVA MODEL--------------------------------')
y_pred = np.ones(labels_ini.shape[0], dtype=np.int) * -1
for i, (idx_train, idx_test) in enumerate(folds):
    print('\nFold %d:' % i)

    y_pred_label = []
    for label in labels_names:
        print('Load %s model:' % label)

        dump_file = './models_nw_dn/' + label + '_best_gs_pipeline.pkl'
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

        class_weight = {}
        if label == 'RESIDENTIAL':
            class_weight[-1] = class_weights['RESIDENTIAL']
            class_weight[1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])
        else:
            class_weight[1] = class_weights[label]
            class_weight[-1] = np.sum([class_weights[value] for value in class_weights.keys() if value != label])

        sample_weights_bin = np.array([class_weight[i] for i in labels[idx_test]])

        print('Training...')
        model.fit(data.iloc[idx_train], labels[idx_train])

        print('Predicting...')
        pred_proba = model.predict_proba(data.iloc[idx_test])
        pred = model.predict(data.iloc[idx_test])

        print('Local clasification report:')
        print('Normal:')
        print(classification_report(labels[idx_test], pred))
        print('Weigthed:')
        print(classification_report(labels[idx_test], pred, sample_weight=sample_weights_bin))

        if label != 'RESIDENTIAL':
            y_pred_label.append(pred_proba[:, 1])
        else:
            y_pred_label.append(pred_proba[:, 0])

    ########################################### Naive Bayes #############################################

    y_pred_label = np.array(y_pred_label).T
    best_params = {'var_smoothing': 8.367363265306123e-05}
    nb = GaussianNB(**best_params)
    nb.fit(data.iloc[idx_train], labels_ini[idx_train])

    y_pred_nb = nb.predict_proba(data.iloc[idx_test])

    nrows, ncols = y_pred_label.shape

    # # Add >0.5 column
    y_pred_label = np.concatenate([y_pred_label, np.ones((nrows, 1))], axis=1)
    y_pred_nb = np.concatenate([y_pred_nb, np.zeros((nrows, 1))], axis=1)

    # Update nrows, ncols
    nrows, ncols = y_pred_label.shape

    # Sort m_ord
    y_pred_nb_argsort = y_pred_nb.argsort(axis=1)[:, ::-1]

    # m_prob sorted by m_ord
    y_pred_label_ord = y_pred_label[(np.arange(nrows).repeat(ncols), y_pred_nb_argsort.ravel())].reshape(nrows, ncols)
    # roll back to m_prob indexes
    y_pred_i = y_pred_nb_argsort[np.arange(nrows), (y_pred_label_ord > 0.5).argmax(1)]

    # if all < 0.5 -> greatest col
    if (y_pred_i == 7).any():
        y_pred_i[y_pred_i==7] = np.argmax(y_pred_label[y_pred_i == 7, :7], axis=1)

    y_pred[idx_test] = y_pred_i

    #####################################################################################################

    print('Fold classification report:')
    print('Normal')
    print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]]))
    print('Weigthed:')
    print(classification_report(labels_ini.iloc[idx_test], labels_names[y_pred[idx_test]],
                                sample_weight=sample_weight.iloc[idx_test]))

assert -1 not in np.unique(y_pred)

print('Global classification report:')
print('Normal:')
print(classification_report(labels_ini, labels_names[y_pred]))
print('Weigthed:')
print(classification_report(labels_ini, labels_names[y_pred], sample_weight=sample_weight))
