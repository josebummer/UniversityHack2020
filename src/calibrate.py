import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from scipy.optimize import differential_evolution

import progressbar

#
# Load data
#

# Load data
X = pd.read_csv('data/train_mask_RESIDENTIAL_ONLY.txt',sep='|',index_col='ID')
y = X.iloc[:,-1]
X = X.drop(columns=[y.name])
cols = X.columns

le = LabelEncoder()
le.classes_ = cols.values

X = X.values
y = le.transform(y)
yp = X.argmax(axis=1)


#
# Calculate weights
#

yp_ts = le.transform(pd.read_csv('predictions/Minsait_UniversidadGranada_CodeDigger_1.txt',sep='|').iloc[:,1])

# Calc cv compensation ratio
_,c_true = np.unique(y,return_counts=True)
_,c_pred = np.unique(yp,return_counts=True)

cv_ratio = c_pred/c_true

# Calc test ratio
_,c_pred_test = np.unique(yp_ts,return_counts=True)
adjusted_c_test = c_pred_test/cv_ratio
# adjusted_c_test = c_pred_test.astype(np.float)

factor = 3180/5618 #guessed residential from plot / total test

# Estimate
class_weight = np.delete(adjusted_c_test,5)
class_weight *= factor/class_weight.sum()
class_weight = np.insert(class_weight,5,1-factor)

print(class_weight/c_true)

y_weights = (class_weight/c_true)[y]

#
# Uncalibrated score
#

print(classification_report(y,X.argmax(1),target_names=le.classes_,digits=3,sample_weight=y_weights))

#
# Calibrate prob
#

def predict2(i,X):
    return np.argmax(X+i.reshape(1,-1),axis=1)

def eval_sol2(i,X,y,y_weights):
    return -accuracy_score(y,predict2(i,X),sample_weight=y_weights)

cv = list(StratifiedKFold(shuffle=True, random_state=42).split(X,y))

ypred2 = np.zeros(y.shape)
for tr_idx,ts_idx in progressbar.progressbar(cv):
    de = differential_evolution(
        eval_sol2,
        [(-1,1)]*len(le.classes_),
        popsize=30,
        tol=1e-4,
        workers=-1,
        updating='deferred',
        args=[X[tr_idx],y[tr_idx],y_weights[tr_idx]])
    ypred2[ts_idx] = predict2(de.x,X[ts_idx])

print(classification_report(y,ypred2,target_names=le.classes_,digits=3,sample_weight=y_weights))