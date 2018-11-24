import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def transform_datetime_features(df, holidays_file='holidays.csv'):
    """extract datetime features"""

    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]

    holidays = pd.read_csv(holidays_file, squeeze=True)
    holidays = pd.to_datetime(holidays)

    for col_name in datetime_columns:
        if len(datetime_columns) < 10:
            df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
            df['number_month_{}'.format(col_name)] = df[col_name].dt.month
            df['number_day_{}'.format(col_name)] = df[col_name].dt.day
            df['number_hour_{}'.format(col_name)] = df[col_name].dt.hour
            df['number_hour_of_week_{}'.format(col_name)] = df[col_name].dt.hour + df[col_name].dt.weekday * 24
            df['number_minute_of_day_{}'.format(col_name)] = df[col_name].dt.minute + df[col_name].dt.hour * 60
            df['number_holiday_{}'.format(col_name)] = df[col_name].isin(holidays).astype('float16')
        else:
            df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday.astype('float16')
            df['number_month_{}'.format(col_name)] = df[col_name].dt.month.astype('float16')
            df['number_day_{}'.format(col_name)] = df[col_name].dt.day.astype('float16')
            df['number_hour_{}'.format(col_name)] = df[col_name].dt.hour.astype('float16')
            df['number_holiday_{}'.format(col_name)] = df[col_name].isin(holidays).astype('float16')

    return df


def drop_const_cols(df):
    """drop constant columns"""

    constant_columns = [
        col_name
        for col_name in df.columns
        if df[col_name].nunique() == 1
        ]
    df.drop(constant_columns, axis=1, inplace=True)

    return df


def count_encoding(df, categorical_values=None):
    """count encoding of categorical features"""

    # train stage
    if categorical_values is None:
        categorical_values = {}
        for col_name in list(df.columns):
                if col_name.startswith('id') or col_name.startswith('string'):
                    categorical_values[col_name] = df[col_name].value_counts().to_dict()
                    df['count_{}'.format(col_name)] = df[col_name] \
                        .map(lambda x: categorical_values[col_name].get(x, 0))
        return df, categorical_values

    # test stage
    else:
        for col_name in list(df.columns):
            if col_name in categorical_values:
                df['count_{}'.format(col_name)] = df[col_name] \
                    .map(lambda x: categorical_values[col_name].get(x, 0))
        return df


def filter_columns(df, groups=['number']):
    """filter columns to use in model"""

    used_columns = []
    for gr in groups:
        used_columns += [col_name for col_name in df.columns
                        if col_name.startswith(gr)]
    cols_to_drop = df.columns[~df.columns.isin(used_columns)]
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df, used_columns


def std_scaler(df, scaler_mean=None, scaler_std=None):
    """standard scaler"""

    # train stage
    if scaler_mean is None:

        scaler_mean = {}
        scaler_std = {}
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col]-mean)/std
            scaler_mean[col] = mean
            scaler_std[col] = std

        return df, scaler_mean, scaler_std

    # test stage
    else:

        for col in df.columns:
            df[col] = (df[col]-scaler_mean[col])/scaler_std[col]

        return df


def compute_likelihood(train_fold, test_fold, feature, target, global_bias=30):
    """computing likelihood for one fold"""

    global_avg = train_fold[target].mean()
    agg = train_fold.groupby(feature)[target].agg(['sum', 'count'])
    values = ((agg['sum'] + global_bias * global_avg) / (agg['count'] + global_bias)).to_dict()

    return test_fold[feature].map(lambda x: values.get(x, global_avg)).values


def likelihood_encoding(df, cat_cols, target='target', categorical_values=None,
                        global_avg=None, global_bias=30, n_folds=5, drop_original=False):

    """likelihood (mean target) encoding"""

    # train stage
    if categorical_values is None:

        categorical_values = {}
        global_avg = {}

        for feature in cat_cols:

            kf = KFold(n_folds, shuffle=True, random_state=123)
            likelihood = pd.Series(index=df.index)
            for train_index, test_index in kf.split(df):
                train_fold = df[[feature, target]].iloc[train_index, :]
                test_fold = df[[feature]].iloc[test_index, :]
                likelihood.iloc[test_index] = compute_likelihood(train_fold,
                    test_fold, feature, target, global_bias)

            if drop_original:
                df[feature] = likelihood
            else:
                df['likelihood_' + feature] = likelihood

            global_avg[feature] = df[target].mean()
            agg = df.groupby(feature)[target].agg(['sum', 'count'])
            categorical_values[feature] = \
                (agg['sum'] + global_bias * global_avg[feature]) / (agg['count'] + global_bias)
            categorical_values[feature] = categorical_values[feature].to_dict()

        return df, categorical_values, global_avg

    # test stage
    else:

        for col_name in list(df.columns):
            if col_name in categorical_values:
                if drop_original:
                    df[col_name] = df[col_name] \
                        .map(lambda x: categorical_values[col_name].get(x, global_avg[col_name]))
                else:
                    df['likelihood_{}'.format(col_name)] = df[col_name] \
                        .map(lambda x: categorical_values[col_name].get(x, global_avg[col_name]))

        return df


def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))
