import argparse
import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

from preprocess import preprocess

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load config
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read data
    df = pd.read_csv(args.test_csv, dtype=model_config['dtypes'],
                     parse_dates=model_config['datetime_cols'])
    print('Dataset read, shape {}'.format(df.shape))
    print('time elapsed: {}'.format(time.time()-start_time))

    # preprocessing
    if model_config['is_big'] or model_config['is_small'] or not model_config['mean_target']:
        df, df_pred = preprocess(df, model_config, type='test')
    else:
        df, df_pred = preprocess(df, model_config,
                                      likelihood_enc=True, type='test')
    print('time elapsed: {}'.format(time.time()-start_time))

    # final data shape
    print('final df shape {}'.format(df.shape))

    # make prediction
    if model_config['is_small']:

        df.fillna(-1, inplace=True)
        model_files = os.listdir(model_config['model_path'])
        pred = []
        for i, f in enumerate(model_files):
            model = joblib.load(os.path.join(model_config['model_path'], f))
            if model_config['mode'] == 'regression':
                pred.append(model.predict(df))
            elif model_config['mode'] == 'classification':
                pred.append(model.predict_proba(df)[:, 1])
        pred = np.array(pred)
        pred = pred.mean(axis=0)

    else:

        preds = []
        for model in model_config['models']:
            if model_config['mode'] == 'regression':
                preds.append(model.predict(df))
            elif model_config['mode'] == 'classification':
                preds.append(model.predict_proba(df)[:, 1])
        pred = np.zeros_like(preds[0])
        for i, p in enumerate(preds):
            pred += p*model_config['model_weights'][i]

    if model_config['positive_target']:
        pred[pred<0] = 0

    df_pred['prediction'] = pred
    df_pred[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
