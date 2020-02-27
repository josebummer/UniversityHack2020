
import pandas as pd
import numpy as np
from collections import Counter

train = pd.read_csv('./data/train.txt', sep='|')
labels = train.CLASE
test = pd.read_csv('./data/test.txt', sep='|')

train = train.drop(['ID'], axis=1)
train['CADASTRALQUALITYID'] = train['CADASTRALQUALITYID'].astype('category')

train = train.drop(['CLASE'], axis=1)
for column in train:
    if train[column].dtype.name == 'category':
        train[column].cat.add_categories('Unknown', inplace=True)
        train[column].fillna('Unknown', inplace=True)
    elif train[column].dtype.name != 'category':
        train[column].fillna(train[column].median(), inplace=True)
train_y = pd.concat([train,labels], axis=1)

def correspondencia_valor_clase(train_y, var):
    # Porcentajes:

    clases = train_y.CLASE.unique()
    values = train_y[var].unique()

    porcentajes = {}
    for v in values:
        total = train_y.loc[(train_y.loc[:, var] == v)].shape[0]

        aux = []
        for c in clases:
            # print(col_name,' , ',v,' , ',c,' , ',round(((train_y.loc[(train_y.loc[:,col_name] == v) & (train_y.loc[:,'status_group'] == c)].shape[0])/total)*100,2), '%')
            aux.append(round(
                ((train_y.loc[(train_y.loc[:, var] == v) & (train_y.loc[:, 'CLASE'] == c)].shape[0]) / total) * 100, 2))
        porcentajes[v] = aux

    # Diferencias:

    diferencias = {}
    for i in range(0, values.shape[0]):
        if i < (values.shape[0] - 1):
            for j in range(i + 1, values.shape[0]):
                name = values[i] + ' - ' + values[j]
                diferencias[name] = round(float(sum(
                    abs(pd.DataFrame(porcentajes[values[i]]).values - pd.DataFrame(porcentajes[values[j]]).values))), 2)

    return porcentajes, diferencias

p,d = correspondencia_valor_clase(train_y,'CADASTRALQUALITYID')