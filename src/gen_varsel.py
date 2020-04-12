import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import time

import threading

from itertools import chain

from expermientos1 import prepare_data, fillna, to_numeric

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label','-l',dest='label',required=True)
parser.add_argument('--threads','-t',dest='n_jobs', type=int, required=True)
parser.add_argument('--popsize','-n',dest='popsize', type=int, default=10)
parser.add_argument('--repeat','-r',dest='repeat', type=int, default=1)
args = parser.parse_args()

N_JOBS = args.n_jobs
POP_SIZE = args.popsize
TARGET_LABEL = args.label
REPEAT_SPLIT = args.repeat

def best_models_and_labels():
    train = pd.read_csv('data/train.txt', sep='|', index_col='ID')

    labels_ini = train.iloc[:, -1]

    labels_names = np.unique(labels_ini)

    models = {}
    labels = {}
    for label in labels_names:
        print('Load %s model:' % label)

        dump_file = './1models_nw_dn/' + label + '_best_gs_pipeline.pkl'
        with open(dump_file, 'rb') as ofile:
            grid = pickle.load(ofile)

        model = grid.best_estimator_
        for step in model.steps:
            if step[0] in ['enn', 'clf']:
                step[1].n_jobs = -1

        if label != 'RESIDENTIAL':
            y_train = np.array([1 if x == label else -1 for x in labels_ini])
        else:
            y_train = np.array([-1 if x == label else 1 for x in labels_ini])

        models[label] = (model)
        labels[label] = (y_train)

    return models, labels


#
# ALGORITMO GENÉTICOk
#

from sklearn.base import clone

CACHE = {}

class de_problem:
    def __init__(self,m,X,y,cvs,cache=CACHE):
        self.m = clone(m)
        self.X = np.array(X)
        self.y = np.array(y)
        self.cvs = cvs
        self.mem = cache

        self.n_cols = X.shape[1]
        self.base_p = np.power(2, np.arange(0, self.n_cols))

    def hash_bool(self,x):
        return self.base_p[x].sum()

    def get_bounds(self):
        return ([0] * self.n_cols, [1] * self.n_cols)

    def get_nix(self):
        return self.n_cols

    def fitness(self,msk):
        #transform 1./0. float to binary
        msk = msk.astype(np.bool)
        x = self.X
        y = self.y

        # Memorize
        x_hash = self.hash_bool(msk)
        if x_hash in self.mem:
            f =  self.mem[x_hash]
        # Get f
        else:
            # Update model threads
            n_jobs = None
            if multiprocessing.current_process().name == 'MainProcess':
                n_jobs = N_JOBS
            else:
                n_jobs = 1

            for l in self.m.steps:
                if 'n_jobs' in l[1].get_params():
                    # print("L",l)
                    l[1].set_params(**{'n_jobs': n_jobs})

            # Splits predict
            y_target = []
            y_predict = []
            for tr_idx,ts_idx in self.cvs:
                self.m.fit(x[tr_idx][:,msk], y[tr_idx])
                yp = self.m.predict(x[ts_idx][:,msk])
                y_target.append(y[ts_idx])
                y_predict.append(yp)

            f = -f1_score(np.concatenate(y_target),np.concatenate(y_predict), average='macro')
            self.mem[x_hash] = f

        return (f,)


class bin_estationary:
    def __init__(self, gen=1, m=0.02):
        self.gen = gen
        self.m = m
        self.evolve_n = 0
        self.verbosity = np.inf

    def set_verbosity(self,l):
        self.verbosity = l


    def evolve(self, pop):
        peval = pop.problem.fitness

        for _ in range(self.gen):
            # Select two parents
            X = pop.get_x()
            f = pop.get_f()

            i, j = np.random.choice(
                range(X.shape[0]),
                size=2,
                replace=False)

            # Cruce
            unif_msk = np.random.randint(0, 2, X.shape[1])
            X_p = X[[i, j]].copy()
            X_p[0, unif_msk], X_p[1, unif_msk] = X_p[1, unif_msk], X_p[0, unif_msk]

            # Mutation
            mut_msk = np.random.choice(
                [0, 1],
                size=X_p.shape,
                p=[1 - self.m, self.m]).astype(np.bool)

            X_p[mut_msk] = np.mod(X_p[mut_msk] + 1, 2)

            # Eval
            f_p = [peval(X_p[i]) for i in range(2)]

            # Replace
            best_i = np.argmin(f_p)
            worst_i = pop.worst_idx()

            if f_p[best_i] < f[worst_i]:
                pop.set_xf(worst_i, X_p[best_i], f_p[best_i])

            self.evolve_n += 1
            if (self.evolve_n % self.verbosity) == 0:
                print(f'Iteration {self.evolve_n}: {pop.champion_f}')

        return pop

import multiprocessing
import json
# import threading
import pygmo as pg


def gen_varsel(m,x,y,cvs):
    # 0.88 minutos por gen
    MERGE_EVERY = 6
    SAVE_EVERY = 1
    MAX_EVOLVE_N = 200

    ALG_ITERS = int(np.gcd(SAVE_EVERY, MAX_EVOLVE_N))
    counter = 0

    N_ISLANDS = N_JOBS
    N_ISLANDS = N_ISLANDS if N_ISLANDS > 0 else multiprocessing.cpu_count()
    print("Procesadores: ", N_ISLANDS)

    #SEED = config_dict['seed']

    print(f'Merge freq: {MERGE_EVERY} | Evolve for: {MAX_EVOLVE_N}')


    prob = pg.problem(de_problem(m,x,y,cvs))
    algo = pg.algorithm(bin_estationary(gen= MERGE_EVERY, m=0.1))
    algo.set_verbosity(1)

    archi = pg.archipelago(
        N_ISLANDS,
        t=pg.ring(),
        algo=algo,
        prob=prob,
        pop_size=POP_SIZE,
        udi=pg.mp_island(),
        seed=1)

    while (counter < MAX_EVOLVE_N):
        counter += ALG_ITERS

        archi.evolve(ALG_ITERS)
        archi.wait()

        every_x = [isl.get_population().get_x().tolist() for isl in archi]
        every_f = [isl.get_population().get_f().tolist() for isl in archi]

        #is cold?
        np_x = np.array(every_x)
        coldness = np_x.reshape((-1, np_x.shape[-1])).std(0).mean()
        is_cold = coldness < 1e-2

        if ((counter % SAVE_EVERY) == 0) or is_cold:
            # print("COLDNESS", coldness)
            # print(every_f)

            with open(f'GEN_OUTPUT_{TARGET_LABEL}.json','w') as ofile:
                json.dump({
                    "best_x": [isl.get_population().champion_x.tolist() for isl in archi],
                    "best_f": [isl.get_population().champion_f.tolist() for isl in archi],
                    "every_x": every_x,
                    "every_f": every_f
                },ofile)

        if is_cold:
            break
#
# if __name__ == "__main__":
#     main()

# %%


def enn_transform(m,x,y,splits):
    xy_splits = []

    scaler = m.steps[0][1]
    enn = m.steps[1][1]
    m = m.steps[2][1]

    for tr_idx, ts_idx in splits:

        # Scaler
        scaler.fit(x[tr_idx])
        x_tr = scaler.transform(x[tr_idx])
        x_ts = scaler.transform(x[ts_idx])

        y_tr = y[tr_idx]
        y_ts = y[ts_idx]

        # ENN
        x_tr, y_tr = enn.fit_resample(x_tr,y_tr)

        # Append
        xy_splits.append([x_tr,x_ts,y_tr,y_ts])

    return m,xy_splits


def main():
    #Si se usa train hay que llamar a los métodos de experimentos1 que se importan arriba.
    #Si se hace con trainGuille no hace falta.

    data = pd.read_csv('data/train.txt', sep='|', index_col='ID')
    # labels_ini = data.iloc[:, -1]
    data.drop('CLASE', axis=1, inplace=True)

    data = prepare_data(data)
    data = fillna(data)
    data = to_numeric(data)

    models, labels = best_models_and_labels()

    mod = models[TARGET_LABEL]


    x = data.to_numpy()
    y = labels[TARGET_LABEL]

    split = StratifiedShuffleSplit(n_splits=REPEAT_SPLIT,test_size=1/3,random_state=42)
    cvs = list(split.split(x,y))

    gen_varsel(mod,x,y,cvs)

    # mod, xy_splits = enn_transform(mod,x,y,splits)
    # mod.set_params(**{'n_jobs':1})
    # x_tr, x_ts, y_tr, y_ts = xy_splits[0]

    # beg = time.time()
    # mod.fit(x_tr,y_tr)
    # print(classification_report(y_ts,mod.predict(x_ts)))
    # elapsed = time.time()-beg
    # print(f'time: {elapsed:.4f}')



if __name__ == '__main__':
    main()