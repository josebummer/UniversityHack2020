import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from scipy.optimize import differential_evolution

import progressbar

final_text = []

#
# Load data
#

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--in','-i',dest='in_template',required=True)
args = parser.parse_args()

# Load data

# IN_TEMPLATE = "data/{}_RGBA+GEOM.txt"
IN_TEMPLATE = args.in_template
IN_PATH = IN_TEMPLATE.format("train")
IN_PATH_TS = IN_TEMPLATE.format("test")


le = LabelEncoder()
le.classes_ = np.array(['AGRICULTURE', 'INDUSTRIAL', 'OFFICE', 'OTHER', 'PUBLIC',
       'RESIDENTIAL', 'RETAIL'])


X = pd.read_csv(IN_PATH,sep='|',index_col='ID')
y = X.CLASE
X = X.drop(columns=[y.name])
cols = X.columns

val_idx = X.idx_test.astype(np.int)
cv = [((val_idx!=i).values.nonzero(),(val_idx==i).values.nonzero()) for i in range(5)]

X = X
y = le.transform(y)

#
# Calculate weights
#
clase_weights = np.array([6.26193840e-05, 4.64739874e-05, 4.55873618e-05, 3.83198034e-05, 3.78393795e-05, 4.81255214e-06, 4.26270960e-05])
y_weights = clase_weights[y].ravel()

#
# Uncalibrated score
#

X_uncalib = np.zeros((X.shape[0],7))
for tr_idx, ts_idx in cv:
    cv_cols = [f'CV_{i}_{l}' for l in le.classes_ for i in [0]]
    X_uncalib[ts_idx] = X[cv_cols].iloc[ts_idx]

cr = classification_report(y,X_uncalib.argmax(1),target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# Without calibration",cr]
#
# Calibrate prob
#

from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score


def predict2(i, X):
    return np.argmax(X + i.reshape(1, -1), axis=1)


def eval_sol2(i, X, y, y_weights):
    return -accuracy_score(y, predict2(i, X), sample_weight=y_weights)

xs = []
yp_prob_calib = np.zeros(y.size).astype(np.int)

for tr_idx, ts_idx in cv:
    cv_cols = [f'CV_{i}_{l}' for l in le.classes_ for i in [0]]
    yp_prob = X[cv_cols].values
    yp_prob = X_uncalib

    de = differential_evolution(
        eval_sol2,
        [(-1, 1)] * len(le.classes_),
        popsize=30,
        tol=1e-4,
        workers=-1,
        updating='deferred',
        args=[yp_prob[tr_idx], y[tr_idx], y_weights[tr_idx]])

    xs.append(de.x)
    yp_prob_calib[ts_idx] = predict2(de.x, yp_prob[ts_idx])


print(xs)
final_text+= ["# Xs",xs]

cr = classification_report(y,yp_prob_calib,target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# With calibration",cr]

xs = np.array(xs)
xs = xs - np.mean(xs,axis=1,keepdims=True)
xs_mean = xs.mean(0)
xs_std = xs.std(0)
print("# StDev",xs_std)
final_text+= ["# StDev",xs_std]
print("# Mean", xs_mean)
final_text+= ["# Mean",xs_mean]

ypred3 = predict2(xs_mean,X_uncalib)
cr = classification_report(y,ypred3,target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# With mean calibration",cr]

# #
# # Test prediction
# #
#
# X = pd.read_csv(IN_PATH_TS,sep='|',index_col='ID')
# ts_indices = X.index
# cols = X.columns
#
# X = X.values
# yp = predict2(xs_mean,X)
# yp_labels = le.inverse_transform(yp)
# df = pd.DataFrame(yp_labels[:,None], index=ts_indices, columns=["CLASE"])
# df.to_csv(f'{IN_PATH_TS}_calibrated_out.csv',sep='|')


with open(f'{IN_PATH}_result.txt','w') as ofile:
    ofile.write("\n".join(map(str,final_text+[""])))