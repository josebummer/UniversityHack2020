from sklearn.naive_bayes import GaussianNB
from expermientos1 import prepare_data, fillna, to_numeric
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report

data = pd.read_csv('/home/jose/Escritorio/datathon/src/data/train.txt', sep='|', index_col='ID')
OVApredictions = pd.read_csv('/home/jose/Escritorio/datathon/src/data/trainOVA.txt', sep='|', index_col='ID')

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

X_train, X_test, y_train, y_test = train_test_split(data, labels_ini, test_size=0.2, random_state=42)

best_params = {'var_smoothing': 8.367363265306123e-05}
pipe_nb = Pipeline([('scl', StandardScaler()),
                    ('clf', GaussianNB(**best_params))])

labels_names = np.unique(labels_ini)

pipe_nb.fit(X_train, y_train)

y_pred = pipe_nb.predict_proba(X_test)

OVAsum = []
for row in OVApredictions.values:
    x = row[:-1]
    OVAsum.append([(np.sum(x*-1) + (2*x[i])) for i in range(len(x))])

print()

