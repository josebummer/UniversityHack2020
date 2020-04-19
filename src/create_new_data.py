import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np

from expermientos1 import prepare_data, fillna, to_numeric
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    test = pd.read_csv('data/test.txt', sep='|', index_col='ID')

    labels_ini = train.iloc[:, -1]
    train.drop('CLASE', axis=1, inplace=True)

    data = pd.concat([train,test], sort=False)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    #Eliminamos los colores y los sustituimos por sus medias
    # data['RED'] = data.iloc[:, 2:13].mean(1)
    # data['GREEN'] = data.iloc[:, 13:24].mean(1)
    # data['BLUE'] = data.iloc[:, 24:35].mean(1)
    # data['NIR'] = data.iloc[:, 35:46].mean(1)

    pca = PCA(n_components=8)
    color = pca.fit_transform(data.iloc[:, 2:46])

    data.drop(columns=data.columns[2:46], axis=1, inplace=True)
    data = pd.concat([data, pd.DataFrame(color, index=data.index)], axis=1, sort=False)

    #Creamos las variables nuevas
    # points = [(2.207524e9,165.5605e6), (2.207524e9,165.5605e6), (2.170449e9, 166.0036e6),
    #           (2.205824e9, 166.3199e6), (2.250042e9, 166.2673e6), (2.270527e9, 165.9025e6),
    #           (2.274459e9, 165.5947e6), (2.269886e9, 165.3261e6), (2.211719e9, 165.1699e6),
    #           (2.156419e9, 165.2959e6), (2.142472e9, 165.4747e6), (2.141374e9, 165.8068e6),
    #           (2.166906e9, 165.7316e6), (2.187454e9, 165.4168e6), (2.174702e9, 165.481e6),
    #           (2.202014e9, 165.5483e6), (2.215004e9, 165.4046e6), (2.196768e9, 165.4717e6),
    #           (2.236186e9, 165.4013e6), (2.220204e9, 165.4714e6), (2.219742e9, 165.8038e6)]
    #
    # distances = [np.linalg.norm(data[['X','Y']].values-b, axis=1) for b in points]
    #
    # for i, _ in enumerate(points):
    #     col = 'C_'+str(i)
    #     data[col] = distances[i]

    data['VALUE'] = data['AREA']*data['CADASTRALQUALITYID']*data['MAXBUILDINGFLOOR']
    data['VALUE2'] = data['AREA'] * data['CADASTRALQUALITYID']
    data['VALUE3'] = data['CADASTRALQUALITYID'] * data['MAXBUILDINGFLOOR']
    data['VALUE4'] = data['AREA'] * data['MAXBUILDINGFLOOR']

    #Creacion de grupos
    # dist1 = []
    #
    # for i, xy in enumerate(data[['X','Y']].values):
    #     dist = np.linalg.norm(xy - data.iloc[i+1:, [0,1]].values, axis=1)
    #     # dist[i] = np.max(dist)+1
    #
    #     dist1.append(np.min(dist))

    #TODO la expansion de la etiqueta no está bien, solo se traspasa a los nodos cercanos al que mira y no a todos.

    mask = np.zeros(data.shape[0], dtype=np.bool)
    et = 0
    values = np.empty(data.shape[0], dtype=np.int)
    #TODO hacer un for que recorra todos los elementos y si ya tiene etiqueta la expande y si no la crea
    while not mask.all():
        idx_i = np.argmin(mask)
        distances = np.linalg.norm(data[['X', 'Y']].values - data.iloc[idx_i, [0,1]].values, axis=1)
        distances[idx_i] = np.max(distances)+1
        idx = np.where(distances<=1.5*np.min(distances))[0]
        idx = idx[~mask[idx]]

        values[idx] = et
        values[idx_i] = et

        et += 1
        mask[idx] = True
        mask[idx_i] = True
        # dist1[idx_i] = np.max(dist1)+1

    train, test = data.iloc[:train.shape[0], ], data.iloc[train.shape[0]:, ]
    train = pd.concat([train,labels_ini], axis=1, sort=False)

    train.to_csv('data/new_train.txt', sep='|')
    test.to_csv('data/new_test.txt', sep='|')

if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    main()