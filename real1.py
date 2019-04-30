import numpy as np
import os
import matplotlib.pyplot as plt

import sklearn.datasets as skld
import sklearn.model_selection as sklms
import sklearn.ensemble as skle

from bayeso import bo
from bayeso.utils import utils_bo
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting
from bayeso import constants

STR_FUN_TARGET = 'real1'
INT_BO = 20
INT_ITER = 50
INT_INIT = 3

data = skld.fetch_olivetti_faces()
images = data.images
targets = data.target

print(images.shape)
print(targets.shape)

X_train, X_test, y_train, y_test = sklms.train_test_split(images, targets, test_size=0.2, random_state=42, stratify=targets)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

def fun_target(X):
    print(X)
    model_ = skle.RandomForestClassifier(n_estimators=int(X[0]), max_depth=int(X[1]), min_samples_split=int(X[2]), max_features=float(X[3]), random_state=42)
    model_.fit(X_train, y_train)
    y_pred = model_.predict(X_test)

    score_ = 1.0 - np.mean(y_pred == y_test)
    print(score_)
    return score_

def main(str_optimizer_method_gp, str_mlm_method, str_ms_method, int_bo, int_iter, int_init):    
    bounds = np.array([
        [10, 100],
        [2.01, 5],
        [2.01, 6],
        [0.001, 1],
    ])

    list_Y = []
    list_time = []
    for ind_bo in range(0, int_bo):
        print('BO Iteration', ind_bo)
        model_bo = bo.BO(bounds, str_optimizer_method_gp=str_optimizer_method_gp, debug=False)
        X_final, Y_final, time_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter, str_initial_method_bo='uniform', str_initial_method_ao='uniform', int_samples_ao=100, str_mlm_method=str_mlm_method, str_modelselection_method=str_ms_method, int_seed=77*(ind_bo+1))
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    if int_bo == 1:
        arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    else:
        arr_Y = np.squeeze(arr_Y)
    arr_Y = np.expand_dims(arr_Y, axis=0)
    arr_time = np.array(list_time)
    arr_time = np.expand_dims(arr_time, axis=0)
    print(np.array2string(arr_Y, separator=','))
    print(np.array2string(arr_time, separator=','))
    utils_plotting.plot_minimum(arr_Y, [STR_FUN_TARGET], int_init, True, path_save=None, str_postfix=None)
    utils_plotting.plot_minimum_time(arr_time, arr_Y, [STR_FUN_TARGET], int_init, True, path_save=None, str_postfix=None)

    return arr_Y, arr_time


# regular / ml
str_method = 'BFGS'
str_mlm_method = 'regular'
str_ms_method = 'ml'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

str_method = 'L-BFGS-B'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

# regular / loocv
str_method = 'BFGS'
str_mlm_method = 'regular'
str_ms_method = 'loocv'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}_loocv.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

str_method = 'L-BFGS-B'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}_loocv.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

# converged / ml
str_method = 'BFGS'
str_mlm_method = 'converged'
str_ms_method = 'ml'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}_converged.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

str_method = 'L-BFGS-B'

arr_Y, arr_time = main(str_method, str_mlm_method, str_ms_method, INT_BO, INT_ITER, INT_INIT)
dict_ = {'arr_Y': arr_Y, 'arr_time': arr_time}
np.save('./results/{}_{}_bo_{}_iter_{}_init_{}_converged.npy'.format(STR_FUN_TARGET, str_method, INT_BO, INT_ITER, INT_INIT), dict_)

