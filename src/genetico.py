import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.compose import TransformedTargetRegressor

from imblearn.pipeline import Pipeline

from collections import Counter

from scipy.optimize import differential_evolution

from scipy.interpolate import interp1d

from itertools import accumulate

import pygmo as pg

from imblearn.under_sampling import RandomUnderSampler

import json

## Prepare data

# Open files
X = pd.read_csv('data/trainPreprocessed.csv',index_col='ID')
# tr = pd.read_csv('data/train.csv').drop(columns='id')
# ts = pd.read_csv('data/test.csv').drop(columns='id')

# Drop columns
# cols_to_drop = ['X_1','X_27','X_25']
# tr.drop(columns=cols_to_drop, inplace=True)
# ts.drop(columns=cols_to_drop, inplace=True)

# X,y train / test split
x = X
y = X.pop('CLASE')

le = LabelEncoder()
le.classes_ = np.unique(y)
y = le.transform(y)
y_map = np.array([ 1.5461463 , -1.6605004 , -0.8112606 , -0.2971131 , -0.24273577,
        0.38151917, -0.49217162])
y_reg = y_map[y]

y_count = np.unique(y).size


cw = np.array([6.26193840e-05, 4.64739874e-05, 4.55873618e-05, 3.83198034e-05,
       3.78393795e-05, 4.81255214e-06, 4.26270960e-05])
y_weights = cw[y]*y.size

uy,uc = np.unique(y,return_counts=True)
bcw = cw/cw.sum()
new_c = (bcw*uc/bcw.max()).astype(np.int)

rus = RandomUnderSampler(sampling_strategy=dict(zip(uy,new_c)),random_state=7)
rs_X,rs_ycat = rus.fit_resample(X,y)
rs_idx = rus.sample_indices_
rs_y = y_reg[rs_idx]

x = rs_X
yr = rs_y
yl = rs_ycat

# 5fold CV
cvs = [list(StratifiedKFold(shuffle=True).split(x,yl))]                        # Single
# cvs = [list(StratifiedKFold(5,shuffle=True).split(x,y)) for i in range(5)] # Repeated


from sklearn.base import clone

class de_problem:
    def __init__(self,m,X,yr,yl):
        self.m = clone(m)
        self.X = np.array(X)
        self.yr = np.array(yr)
        self.yl = np.array(yl)
        self.cvs = cvs
        self.mem = {}

        self.n_cols = X.shape[1]
        self.base_p = np.power(2, np.arange(0, self.n_cols))

    def hash_bool(self,x):
        return self.base_p[x].sum()

    def get_bounds(self):
        return ([0] * self.n_cols, [1] * self.n_cols)

    def get_nix(self):
        return self.n_cols

    def fitness(self,x):
        #transform 1./0. float to binary
        x = x.astype(np.bool)

        # Memorize
        x_hash = self.hash_bool(x)
        if x_hash in self.mem:
            f =  self.mem[x_hash]
        # Get f
        else:
            losses = []
            for cv in self.cvs:
                if type(multiprocessing.current_process()) != multiprocessing.Process:
                    yp = cross_val_predict(self.m, self.X[:,x], self.yr, cv=cv, n_jobs=-1)
                else:
                    yp = cross_val_predict(self.m, self.X[:, x], self.yr, cv=cv)
                ypl = np.abs(y_map[None,:]-yp[:,None]).argmin(1)
                loss = f1_score(self.yl,ypl,average='macro')
                losses.append(loss)
            f = np.mean(losses)

        return (-f,)


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
import threading

def is_cool(x):
    pass

def main():
    MERGE_EVERY = 3
    SAVE_EVERY = 1
    MAX_EVOLVE_N = 10000 // MERGE_EVERY

    ALG_ITERS = int(np.gcd(SAVE_EVERY, MAX_EVOLVE_N))
    counter = 0

    N_ISLANDS = 6
    N_ISLANDS = N_ISLANDS if N_ISLANDS > 0 else multiprocessing.cpu_count()
    print("Procesadores: ", N_ISLANDS)

    #SEED = config_dict['seed']

    print(f'Merge freq: {MERGE_EVERY} | Evolve for: {MAX_EVOLVE_N}')

    treg = RandomForestRegressor(n_estimators=100)


    prob = pg.problem(de_problem(treg,x,yr,yl))
    algo = pg.algorithm(bin_estationary(gen= MERGE_EVERY, m=0.1))
    algo.set_verbosity(10)

    archi = pg.archipelago(
        N_ISLANDS,
        t=pg.ring(),
        algo=algo,
        prob=prob,
        pop_size=5,
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
        is_cold = coldness < 1e-10

        if ((counter % SAVE_EVERY) == 0) or is_cold:

            with open(f'GEN_OUTPUT.json','w') as ofile:
                json.dump({
                    "best_x": [isl.get_population().champion_x.tolist() for isl in archi],
                    "best_f": [isl.get_population().champion_f.tolist() for isl in archi],
                    "every_x": every_x,
                    "every_f": every_f
                },ofile)

        if is_cold:
            break

if __name__ == "__main__":
    main()