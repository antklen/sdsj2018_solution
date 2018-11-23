import argparse
import os
import pickle
import time
import pandas as pd
import gc

import warnings
warnings.filterwarnings("ignore")

from preprocess import preprocess
from feature_selection import lgb_importance_fs
from optim_hyperopt import run_hyperopt
from models import lgb_model, xgb_model
from small_data import train_small_data
from sklearn.externals import joblib

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
BIG_DATASET_SIZE = 300 * 1024 * 1024
SMALL_DATA_LEN = 1000

# hyperopt settings
HYPEROPT_NUM_ITERATIONS = 30
HYPEROPT_MAX_TRAIN_SIZE = 300 * 1024 * 1024
HYPEROPT_MAX_TRAIN_ROWS = 5e+6
USE_MEAN_TARGET = True

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()


    # read small amount of data to parse dtypes and find datetime columns
    df0 = pd.read_csv(args.train_csv, nrows=5000)
    dtypes = df0.dtypes.map(lambda x: 'float32' if x=='float64' else x).to_dict()
    datetime_cols = df0.columns[df0.columns.str.contains('datetime')].tolist()
    # read full data with float32 instead of float64 and parsing datetime columns
    df = pd.read_csv(args.train_csv, dtype=dtypes, parse_dates=datetime_cols)

    y = df.target
    df.drop('target', axis=1, inplace=True)
    is_big = df.memory_usage(deep=True).sum() > BIG_DATASET_SIZE
    is_small = df.shape[0] < SMALL_DATA_LEN

    print('Dataset read, shape {}'.format(df.shape))
    print('memory_usage {}'.format(df.memory_usage(deep=True).sum()/1024/1024))
    print('time elapsed: {}'.format(time.time()-start_time))

    # dict with data necessary to make predictions
    model_config = {}
    model_config['is_big'] = is_big
    model_config['is_small'] = is_small
    model_config['mode'] = args.mode
    model_config['dtypes'] = dtypes
    model_config['datetime_cols'] = datetime_cols
    model_config['mean_target'] = USE_MEAN_TARGET

    model_config['positive_target'] = True if y.min() >= 0 else False


    # preprocessing
    if is_big or is_small or not USE_MEAN_TARGET:
        df, model_config = preprocess(df, model_config, type='train')
    else:
        df, model_config = preprocess(df, model_config, y=y,
                                      likelihood_enc=True, type='train')
    print('number of features {}'.format(len(model_config['used_columns'])))
    print('time elapsed: {}'.format(time.time()-start_time))

    gc.collect()

    # feature selection
    if is_big or len(model_config['used_columns']) > 1000:
        df, used_columns = lgb_importance_fs(df, y, args.mode, BIG_DATASET_SIZE)
        model_config['used_columns'] = used_columns
        print('time elapsed: {}'.format(time.time()-start_time))

    # final data shape
    print('final df shape {}'.format(df.shape))

    if model_config['is_small']:

        elapsed = time.time()-start_time
        df.fillna(-1, inplace=True)
        models = train_small_data(df, y, model_config,
                                    time_limit=int((TIME_LIMIT-elapsed)*0.7),
                                    include_algos=['et', 'rf', 'xgb', 'lgb'],
                                    model_seed=42)
        path = os.path.join(args.model_dir, 'models')
        if not os.path.exists(path):
            os.mkdir(path)
        model_config['model_path'] = os.path.join(path)
        total_size = 0
        for i, model in enumerate(models):
            joblib.dump(model, os.path.join(path, 'model_%d' % i),
                        compress=5)
            total_size += os.path.getsize(os.path.join(path, 'model_%d' % i))/1024/1024
            print('total models size', total_size)
            if total_size > 900:
                break
            if time.time()-start_time > TIME_LIMIT*0.95:
                break

    else:

        # hyperopt
        elapsed = time.time()-start_time
        model_type = 'lgb'# if not is_small else 'rf'
        params, models, w = run_hyperopt(df, y, model_config,
                              N=HYPEROPT_NUM_ITERATIONS,
                              time_limit=int((TIME_LIMIT-elapsed)*0.9),
                              model_type=model_type, blend=True,
                              max_train_size=HYPEROPT_MAX_TRAIN_SIZE,
                              max_train_rows=HYPEROPT_MAX_TRAIN_ROWS)

        model_config['models'] = models
        model_config['model_weights'] = w


    # save config to file
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
