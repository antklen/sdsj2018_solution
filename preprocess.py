from utils import transform_datetime_features
from utils import drop_const_cols, filter_columns, std_scaler
from utils import count_encoding, likelihood_encoding


def preprocess(df, model_config, y=None, type='train', count_enc=True,
               likelihood_enc=False, scaling=False,
               use_groups=['number', 'count', 'likelihood']):
    """preprocessing and feature engineering for input data"""

    print('preprocess data..')

    # extract datetime features
    df = transform_datetime_features(df)
    print('datetime features extracted')

    # categorical count encoding
    if count_enc:
        if type == 'train':
            df, categorical_values = count_encoding(df)
            model_config['categorical_values'] = categorical_values
        elif type == 'test':
            df = count_encoding(df, model_config['categorical_values'])
        print('count encoding of categorical features added')

    # mean target encoding
    if likelihood_enc:
        if type == 'train':
            cat_cols = df.columns[df.columns.str.startswith('string')].tolist()
            df, categorical_values, global_avg = likelihood_encoding(df.join(y), cat_cols)
            model_config['mean_target'] = categorical_values
            model_config['global_avg'] = global_avg
            model_config['cat_cols'] = cat_cols
        elif type == 'test':
            df = likelihood_encoding(df, model_config['cat_cols'],
                                     categorical_values=model_config['mean_target'],
                                     global_avg=model_config['global_avg'])
        print('mean target encoding of categorical features added')

    # drop constant features
    if type == 'train':
        df = drop_const_cols(df)

    # filter columns
    if type == 'train':
        df, used_columns = filter_columns(df, groups=use_groups)
        model_config['used_columns'] = used_columns
    elif type=='test':
        df_pred = df[['line_id']]
        df = df[model_config['used_columns']]

    # scaling
    if scaling:
        if type == 'train':
            df, scaler_mean, scaler_std = std_scaler(df)
            model_config['scaler_mean'] = scaler_mean
            model_config['scaler_std'] = scaler_std
        elif type=='test':
            df = std_scaler(df, model_config['scaler_mean'], model_config['scaler_std'])


    if type == 'train':
        return df, model_config
    else:
        return df, df_pred
