import time
import numpy as np
from hyperopt import hp
from hyperopt.pyll import stochastic
from models import lgb_model, xgb_model, rf_model, et_model

fspace_lgb = {
    'num_leaves': hp.quniform('num_leaves', 5, 50, 1),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'min_child_samples': hp.quniform('min_child_samples', 10, 30, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.2))
}

fspace_xgb = {
    'max_depth': 3 + hp.randint('max_depth', 6),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'min_child_weight': hp.quniform('min_child_weight', 10, 30, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.2))
}


def train_small_data(df, y, model_config, time_limit,
                     include_algos=['et', 'rf', 'lgb', 'xgb'], n_boost=10,
                     model_seed=None, verbose=False):

    start_time = time.time()
    mode = model_config['mode']
    n_boost = 10
    models = []

    if 'et' in include_algos:
        for max_f in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            params = {'n_estimators': 500, 'max_depth': 20, 'max_features': max_f,
                      'n_jobs': -1, 'random_state': model_seed}
            model = et_model(params, mode)
            model.fit(df, y)
            models.append(model)
            if verbose:
                print(params)
            if time.time()-start_time >= time_limit*0.95:
                print('time limit exceeded.')
                return models
        print('et done. total time elapsed {}'.format(time.time()-start_time))

    if 'xgb' in include_algos:
        space = [stochastic.sample(fspace_xgb) for i in range(n_boost)]
        for params in space:
            params.update({'n_estimators': 500, 'random_state': model_seed, 'n_jobs': -1})
            model = xgb_model(params, mode)
            model.fit(df, y)
            models.append(model)
            if verbose:
                print(params)
            if time.time()-start_time >= time_limit*0.95:
                print('time limit exceeded.')
                return models
        print('xgb done. total time elapsed {}'.format(time.time()-start_time))

    if 'lgb' in include_algos:
        space = [stochastic.sample(fspace_lgb) for i in range(n_boost)]
        for params in space:
            params['num_leaves'] = int(params['num_leaves'])
            params['min_child_samples'] = int(params['min_child_samples'])
            params.update({'n_estimators': 500, 'subseq_freq': 1, 'random_state': model_seed, 'n_jobs': -1})
            model = lgb_model(params, mode)
            model.fit(df, y)
            models.append(model)
            if verbose:
                print(params)
            if time.time()-start_time >= time_limit*0.95:
                print('time limit exceeded.')
                return models
        print('lgb done. total time elapsed {}'.format(time.time()-start_time))

    if 'rf' in include_algos:
        for max_f in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            params = {'n_estimators': 500, 'max_depth': 20, 'max_features': max_f,
                      'n_jobs': -1, 'random_state': model_seed}
            model = rf_model(params, mode)
            model.fit(df, y)
            models.append(model)
            if verbose:
                print(params)
            if time.time()-start_time >= time_limit*0.95:
                print('time limit exceeded.')
                return models
        print('rf done. total time elapsed {}'.format(time.time()-start_time))


    return models
