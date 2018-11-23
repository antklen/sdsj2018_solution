import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

def lgb_model(params, mode):

    if mode == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    return model

def xgb_model(params, mode):

    if mode == 'regression':
        model = xgb.XGBRegressor(**params)
    else:
        model = xgb.XGBClassifier(**params)
    return model

def rf_model(params, mode):

    if mode == 'regression':
        model = RandomForestRegressor(**params)
    else:
        model = RandomForestClassifier(**params)
    return model

def et_model(params, mode):

    if mode == 'regression':
        model = ExtraTreesRegressor(**params)
    else:
        model = ExtraTreesClassifier(**params)
    return model
