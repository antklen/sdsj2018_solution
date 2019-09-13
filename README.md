# Sberbank Data Science Journey 2018: AutoML

2nd place solution for [Sberbank Data Science Journey 2018 AutoML competition](https://sdsj.sberbank.ai/2018/ru/contest.html)

main scripts:  
train.py - training <br/>
predict.py - prediction on test data

### Preprocessing

Apart from basic preprocessing (extracting datetime features, encoding categorical variables, drop constant features):
- is_holiday flag for each datetime feature
- mean target encoding for categorical features
- for dealing with memory issues:
  * read small part of data, define data types, read entire data with float32 instead of float64
  * parse datetime while reading

### Machine learning approach

- LightGBM
- Hyperopt for parameter tuning <br/>
after each step check if time limit is not exceeded, then continue
- ensemble (blending) of best models from hyperopt
  * during hyperopt iterations remember all models that were trained
  * choose 5 best models in the end
  * blend them with stepwise blending <br/>
  *Caruana et al. (2004) Ensemble Selection from Libraries of Models*
  
  
### Special cases

- Very small data
  * don't use target encoding (to prevent overfitting)
  * don't optimize parameters at all (to prevent overfitting)
  * run several models (LightGBM, XGBoost, RF, ET) with random parameters and average them
- Very big data
  * simple feature selection from LightGBM feature importance
