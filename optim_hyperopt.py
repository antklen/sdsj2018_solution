import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from models import lgb_model, xgb_model, rf_model
from ensemble import stepwise_blend
from utils import rmse


# feature spaces for algorithms to sample from
fspace_lgb = {
    'num_leaves': hp.quniform('num_leaves', 10, 100, 1),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'min_child_samples': hp.quniform('min_child_samples', 10, 50, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.2)),
    'subsample_freq': 1 + hp.randint('subsample_freq', 5)
}
fspace_xgb = {
    'max_depth': 3 + hp.randint('max_depth', 6),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 50, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.2)),
}
fspace_rf = {
    'max_depth': hp.choice('max_depth', [None,5,10,20]),
    'max_features': hp.choice('max_features', [None,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
}


def run_hyperopt(X, y, model_config, N, time_limit, model_type='lgb',
                 return_preds=False, blend=False,
                 max_train_size=None, max_train_rows=None):

    print('hyperopt..')

    start_time = time.time()

    mode = model_config['mode']

    # make cross-validation instead of single train-validation split for small dataset
    # actually wasn't used in final submission
    cv = True if X.shape[0] < 2000 else False

    # train-test split
    train_size = 0.8
    # restrict size of train set to be not greater than max_train_size
    if max_train_size is not None:
        size_factor = max(1, train_size*X.memory_usage(deep=True).sum()/max_train_size)
    # restrict number of rows in train set to be not greater than max_train_rows
    if max_train_rows is not None:
        rows_factor = max(1, train_size*X.shape[0]/max_train_rows)
    train_size = train_size/max(size_factor, rows_factor)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size, random_state=42)
    print('train shape {}, size {}'.format(Xtrain.shape, Xtrain.memory_usage(deep=True).sum()/1024/1024))

    # define search space
    if model_type == 'lgb':
        fspace = fspace_lgb
    elif model_type == 'xgb':
        fspace = fspace_xgb
    elif model_type == 'rf':
        fspace = fspace_rf

    if blend or return_preds:
        models = []
        preds = []
        scores = []

    # objective function to pass to hyperopt
    def objective(params):

        iteration_start = time.time()

        # print(params)
        params.update({'n_estimators': 500, 'random_state': 42, 'n_jobs': -1})

        # define model
        if model_type == 'lgb':
            params['num_leaves'] = int(params['num_leaves'])
            params['min_child_samples'] = int(params['min_child_samples'])
            model = lgb_model(params, mode)
        elif model_type == 'xgb':
            params['n_estimators'] = 500
            params['tree_method'] = 'hist'
            model = xgb_model(params, mode)
        elif model_type == 'rf':
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
            model = rf_model(params, mode)

        # training and prediction
        if cv:

            kf = KFold(n_splits=5, shuffle=True)
            pred = np.zeros_like(y)

            for i, (train_index, test_index) in enumerate(kf.split(X)):

                # train-validation split
                Xtrain2 = X.iloc[train_index]
                Xtest2 = X.iloc[test_index]
                ytrain2 = y.iloc[train_index]
                ytest2 = y.iloc[test_index]

                model.fit(Xtrain2, ytrain2)
                if mode == 'regression':
                    pred[test_index] = model.predict(Xtest2)
                elif mode == 'classification':
                    pred[test_index] = model.predict_proba(Xtest2)[:, 1]

            if mode == 'regression':
                loss = np.sqrt(mean_squared_error(y, pred))
            elif mode == 'classification':
                loss = -roc_auc_score(y, pred)

            model.fit(X, y)

        else:

            model.fit(Xtrain, ytrain)
            if mode == 'regression':
                pred = model.predict(Xtest)
                loss = np.sqrt(mean_squared_error(ytest, pred))
            elif mode == 'classification':
                pred = model.predict_proba(Xtest)[:, 1]
                loss = -roc_auc_score(ytest, pred)


        if blend or return_preds:
            models.append(model)
            preds.append(pred)
            scores.append(loss)

        iteration_time = time.time()-iteration_start
        print('iteration time %.1f, loss %.5f' % (iteration_time, loss))

        return {'loss': loss, 'status': STATUS_OK,
                'runtime': iteration_time,
                'params': params}


    # object with history of iterations to pass to hyperopt
    trials = Trials()

    # loop over iterations of hyperopt
    for t in range(N):
        best = fmin(fn=objective, space=fspace, algo=tpe.suggest,
                    max_evals=t+1, trials=trials)
        # check if time limit exceeded, then interrupt search
        elapsed = time.time()-start_time
        if elapsed >= time_limit:
            print('time limit exceeded')
            break

    print('best parameters', trials.best_trial['result']['params'])


    if blend:

        print('blending..')
        y_blend = y if cv else ytest

        num_best_models = 5
        best = np.argsort(scores)[:num_best_models]
        models = [models[i] for i in best]
        preds = [preds[i] for i in best]

        if mode =='regression':
            res, w = stepwise_blend(preds, y_blend, rmse, N=20, greater_is_better=False)
        elif mode =='classification':
            res, w = stepwise_blend(preds, y_blend, roc_auc_score, N=20)

        return trials.best_trial['result']['params'], models, w

    if return_preds:

        y_true = y if cv else ytest
        return trials.best_trial['result']['params'], models, preds, y_true

    return trials.best_trial['result']['params']
