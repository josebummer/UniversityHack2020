import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Load data
X = pd.read_csv('../src/data/trainOVA.txt',sep='|',index_col='ID')
y = X.iloc[:,-1]
X = X.drop(columns=[y.name])
cols = X.columns

le = LabelEncoder()
le.classes_ = cols.values

X = X.values
y = le.transform(y)
yp = X.argmax(axis=1)

yp_ts = le.transform(pd.read_csv('../src/predictions/Minsait_UniversidadGranada_CodeDigger_1.txt',sep='|').iloc[:,1])

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

y_weights = (class_weight/c_true)[y]
# y_weights = np.ones(y.shape)/y.size

print(classification_report(y,X.argmax(1),target_names=le.classes_,digits=3,sample_weight=y_weights))