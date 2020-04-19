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
# X = pd.read_csv('data/train_mask_RESIDENTIAL_ONLY.txt',sep='|',index_col='ID')
IN_TEMPLATE = "data/{}_RGBA+GEOM.txt"
IN_TEMPLATE = args.in_template
IN_PATH = IN_TEMPLATE.format("train")
IN_PATH_TS = IN_TEMPLATE.format("test")

X = pd.read_csv(IN_PATH,sep='|',index_col='ID')
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
final_text+= ["# Class weight",class_weight/c_true]

y_weights = (class_weight/c_true)[y]

#
# Uncalibrated score
#

cr = classification_report(y,X.argmax(1),target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# Without calibration",cr]
#
# Calibrate prob
#

def predict2(i,X):
    return np.argmax(X+i.reshape(1,-1),axis=1)

def eval_sol2(i,X,y,y_weights):
    return -accuracy_score(y,predict2(i,X),sample_weight=y_weights)

cv = list(StratifiedKFold(shuffle=True, random_state=42).split(X,y))

ypred2 = np.zeros(y.shape)
xs = []
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
    xs.append(de.x)

print(xs)
final_text+= ["# Xs",xs]

cr = classification_report(y,ypred2,target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# With calibration",cr]

xs = np.array(xs)
xs_mean = xs.mean(0)
xs_std = xs.std(0)
print("# StDev",xs_std)
final_text+= ["# StDev",xs_std]
print("# Mean", xs_mean)
final_text+= ["# Mean",xs_mean]

ypred3 = predict2(xs_mean,X)
cr = classification_report(y,ypred3,target_names=le.classes_,digits=3,sample_weight=y_weights)
print(cr)
final_text+= ["# With mean calibration",cr]

#
# Test prediction
#

X = pd.read_csv(IN_PATH_TS,sep='|',index_col='ID')
ts_indices = X.index
cols = X.columns

X = X.values
yp = predict2(xs_mean,X)
yp_labels = le.inverse_transform(yp)
df = pd.DataFrame(yp_labels[:,None], index=ts_indices, columns=["CLASE"])
df.to_csv(f'{IN_PATH_TS}_calibrated_out.csv',sep='|')


with open(f'{IN_PATH}_result.txt','w') as ofile:
    ofile.write("\n".join(map(str,final_text+[""])))