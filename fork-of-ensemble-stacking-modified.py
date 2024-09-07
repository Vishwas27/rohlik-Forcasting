#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


# In[2]:


base_train_df = pd.read_csv('/kaggle/input/rohlik-orders-forecasting-challenge/train.csv')
base_train_df.head(5)


# In[3]:


base_test_df = pd.read_csv('/kaggle/input/rohlik-orders-forecasting-challenge/test.csv')
base_test_df.head(5)


# In[4]:


# Base features
base_features = base_test_df.drop(columns=['id']).columns
test_id = base_test_df['id']


# In[5]:


# train_df = pd.concat([base_train_df[base_features], base_train_df['orders']], axis=1)
train_df = pd.concat([base_train_df[base_features], base_train_df['orders']], axis=1)
test_df = base_test_df[base_features]


# # Preprocessing

# In[6]:


# Info of train or test datasets
print(train_df.info())
print('='*60)
print(test_df.info())


# In[7]:


# Concat train data + test data
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)


# In[8]:


# Convert the date column to a processable format 
date_start = pd.to_datetime(all_df['date'], errors='coerce').min()

date_col = ['date']
for _col in date_col:
    date_col = pd.to_datetime(all_df[_col], errors='coerce')
    all_df[_col + "_year"] = date_col.dt.year.fillna(-1)
    all_df[_col + "_month"] = date_col.dt.month.fillna(-1)
    all_df[_col + "_day"] = date_col.dt.day.fillna(-1)
    all_df[_col + "_day_of_week"] = date_col.dt.dayofweek.fillna(-1)
    all_df[_col + "_week_of_year"] = date_col.dt.isocalendar().week.fillna(-1)

    all_df[_col + "_num"] = (date_col-date_start).dt.days.fillna(-1)
    #train['numday'] = (train['date']-date_start).dt.days
    all_df[_col + "_day_of_year"] = date_col.dt.dayofyear.fillna(-1)
    all_df[_col + "_day_of_year"] = np.where( (all_df[_col + "_year"]%4==0)&(all_df[_col + "_month"]>2), all_df[_col + "_day_of_year"]-1, all_df[_col + "_day_of_year"])

    all_df[_col + "_quarter"] = date_col.dt.quarter.fillna(-1)
    all_df[_col + "_is_month_start"] = date_col.dt.is_month_start.astype(int).fillna(-1)
    all_df[_col + "_is_month_end"] = date_col.dt.is_month_end.astype(int).fillna(-1)
    all_df[_col + "_is_quarter_start"] = date_col.dt.is_quarter_start.astype(int).fillna(-1)
    all_df[_col + "_is_quarter_end"] = date_col.dt.is_quarter_end.astype(int).fillna(-1)
    # all_df[_col + '_is_weekend'] = all_df['date_day_of_week'].isin([5, 6]).astype(int)
    # all_df.drop(_col, axis=1, inplace=True)

all_df['date'] = pd.to_datetime(all_df['date'])
all_df


# In[9]:


# Apply sine and cosine transformations
#all_df['year_sin'] = all_df['date_year'] * np.sin(2 * pi * all_df['date_year'])
#all_df['year_cos'] = all_df['date_year'] * np.cos(2 * pi * all_df['date_year'])
all_df['month_sin'] = all_df['date_month'] * np.sin(2 * pi * all_df['date_month'])
all_df['month_cos'] = all_df['date_month'] * np.cos(2 * pi * all_df['date_month'])
all_df['day_sin'] = all_df['date_day'] * np.sin(2 * pi * all_df['date_day'])
all_df['day_cos'] = all_df['date_day'] * np.cos(2 * pi * all_df['date_day'])

all_df['year_sin'] = np.sin(2 * pi * all_df["date_day_of_year"])
all_df['year_cos'] = np.cos(2 * pi * all_df['date_day_of_year'])


# In[10]:


# Replace Null values with None
all_df['holiday_name'].fillna('None', inplace=True)


# In[11]:


# OneHotEncoding → holiday_name
enc = OneHotEncoder( sparse=False )

holiday_encoded = enc.fit_transform(all_df[['holiday_name']])
encoded_df = pd.DataFrame(holiday_encoded, columns=enc.get_feature_names_out(['holiday_name']))
all_df = pd.concat([all_df, encoded_df], axis=1)

# drop holiday_name column
all_df = all_df.drop('holiday_name', axis=1)


# In[12]:


# LabelEncoding → warehouse column
le = preprocessing.LabelEncoder()

all_df['warehouse'] = le.fit_transform(all_df['warehouse'])

# holiday_name
# all_df['holiday_name'] = le.fit_transform(all_df['holiday_name'])


# In[13]:


# Obtain the data for the day before or after a holiday
all_df['holiday_before'] = all_df['holiday'].shift(1).fillna(0).astype(int)
all_df['holiday_after'] = all_df['holiday'].shift(-1).fillna(0).astype(int)

# Obtain the data for the day before or after a shops_closed → It did not lead to an improvement in the MAPE score
# all_df['shops_closed_before'] = all_df['shops_closed'].shift(1).fillna(0).astype(int)
# all_df['shops_closed_after'] = all_df['shops_closed'].shift(-1).fillna(0).astype(int)

# Obtain the data for the day before or after school_holidays → It did not lead to an improvement in the MAPE score
# all_df['winter_school_holidays_before'] = all_df['winter_school_holidays'].shift(1).fillna(0).astype(int)
# all_df['winter_school_holidays_after'] = all_df['winter_school_holidays'].shift(-1).fillna(0).astype(int)
# all_df['school_holidays_before'] = all_df['school_holidays'].shift(1).fillna(0).astype(int)
# all_df['school_holidays_after'] = all_df['school_holidays'].shift(-1).fillna(0).astype(int)

# Obtain the data for the day before or after weekends → It did not lead to an improvement in the MAPE score
# all_df['weekend_before'] = all_df['date_is_weekend'].shift(1).fillna(0).astype(int)
# all_df['weekend_after'] = all_df['date_is_weekend'].shift(-1).fillna(0).astype(int)


# In[14]:


# Convert the data back to train_df and test_df
train_df_le = all_df[~all_df['orders'].isnull()]
test_df_le = all_df[all_df['orders'].isnull()]

train_df_le = train_df_le.drop(columns=['date'], axis=1)
test_df_le = test_df_le.drop(columns=['date'], axis=1)


# In[15]:


# # Predict the user_activity_2 using LGBM and add the results to the test dataset → It did not lead to an improvement in the MAPE score
# features = [col for col in train_df_le.columns if col not in ['orders', 'user_activity_2']]
# X_user_activity2 = train_df_le[features]
# y_user_actibity2 = train_df_le['user_activity_2']

# # LGBM for generating the user_activity_2 column
# X_ua2_train, X_ua2_val, y_ua2_train, y_ua2_val = train_test_split(X_user_activity2, y_user_actibity2, test_size=0.2, random_state=42)

# lgb_ua2_train = lgb.Dataset(X_ua2_train, y_ua2_train)
# lgb_ua2_val = lgb.Dataset(X_ua2_val, y_ua2_val, reference=lgb_ua2_train)
# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'verbosity': -1,
#     'seed': 42
# }
# model_user_activity = lgb.train(params, lgb_ua2_train, valid_sets=[lgb_ua2_train, lgb_ua2_val])

# # Predict and fill in the user_activity_2
# test_df_le['user_activity_2'] = model_user_activity.predict(test_df_le[features]).round()


# # Modeling (Ensemble + Stacking)

# **Ensemble**
# * LightGBM 
# * XGBoost 
# * RandomForest 
# * CatBoost
# * Logistic Regression
# * Ada Boost
# * Decision Tree
# * Gradient Boost

# In[16]:


# split train data

# Set random seed 
random_seed = 42 

X = train_df_le.drop(columns=['orders'])
y = train_df_le['orders']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


# In[17]:


# Cross validation

# Number of splits for cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

# Placeholders for stacking features
stacking_train = np.zeros((X_train.shape[0], 8))
stacking_test = np.zeros((X_test.shape[0],8))

# Initialize base models
lgb_model = lgb.LGBMRegressor(random_state=random_seed)
xgb_model = xgb.XGBRegressor(random_state=random_seed)
cat_model = CatBoostRegressor(silent=True, random_state=random_seed)
rf_model = RandomForestRegressor(random_state=random_seed)
lr_model = LogisticRegression(random_state=random_seed)
ad_model = AdaBoostRegressor(random_state=random_seed)
dt_model = DecisionTreeRegressor(random_state=random_seed)
gb_model = GradientBoostingRegressor(random_state=random_seed)

# Train base models with cross-validation
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train each base model
    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    cat_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    lr_model.fit(X_tr, y_tr)
    ad_model.fit(X_tr, y_tr)
    dt_model.fit(X_tr, y_tr)
    gb_model.fit(X_tr, y_tr)

    # Predict on validation set
    stacking_train[val_idx, 0] = lgb_model.predict(X_val)
    stacking_train[val_idx, 1] = xgb_model.predict(X_val)
    stacking_train[val_idx, 2] = cat_model.predict(X_val)
    stacking_train[val_idx, 3] = rf_model.predict(X_val)
    stacking_train[val_idx, 4] = lr_model.predict(X_val)
    stacking_train[val_idx, 5] = ad_model.predict(X_val)
    stacking_train[val_idx, 6] = dt_model.predict(X_val)
    stacking_train[val_idx, 7] = gb_model.predict(X_val)

    # Predict on test set
    stacking_test[:, 0] += lgb_model.predict(X_test) / n_splits
    stacking_test[:, 1] += xgb_model.predict(X_test) / n_splits
    stacking_test[:, 2] += cat_model.predict(X_test) / n_splits
    stacking_test[:, 3] += rf_model.predict(X_test) / n_splits
    stacking_test[:, 4] += lr_model.predict(X_test) / n_splits
    stacking_test[:, 5] += ad_model.predict(X_test) / n_splits
    stacking_test[:, 6] += dt_model.predict(X_test) / n_splits
    stacking_test[:, 7] += gb_model.predict(X_test) / n_splits
    

# Train meta-model
# meta_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
meta_model = LGBMRegressor(n_estimators=180, num_leaves=16, learning_rate=0.06, colsample_bytree=0.6, lambda_l1=0.2, lambda_l2=0.2, random_state=random_seed)
meta_model.fit(stacking_train, y_train)
best_iteration = meta_model.best_iteration_

# Predict on test set using meta-model
meta_pred = meta_model.predict(stacking_test)


# In[18]:


# Evaluate results
mape = mean_absolute_percentage_error(y_test, meta_pred)
print(f'Simple Average Ensemble Model MAPE: {mape:.4f}')
#0.0318


# In[19]:


# Prediction
# test_df_le = test_df_le.drop(columns=['date', 'orders'])
test_df_le = test_df_le.drop(columns=['orders'])

lgb_pred_test = lgb_model.predict(test_df_le)
xgb_pred_test = xgb_model.predict(test_df_le)
cat_pred_test = cat_model.predict(test_df_le)
rf_pred_test = rf_model.predict(test_df_le)
lr_pred_test = lr_model.predict(test_df_le)
ad_pred_test = ad_model.predict(test_df_le)
dt_pred_test = dt_model.predict(test_df_le)
gb_pred_test = gb_model.predict(test_df_le)

# stacking_test_df_le = np.vstack([lgb_pred_test, xgb_pred_test, cat_pred_test, rf_pred_test]).T
stacking_test_df_le = np.vstack([lgb_pred_test, 
                                 xgb_pred_test, 
                                 cat_pred_test, 
                                 rf_pred_test, 
                                 lr_pred_test, 
                                 ad_pred_test, 
                                 dt_pred_test, 
                                 gb_pred_test
                                ]).T

submit_pred = meta_model.predict(stacking_test_df_le)


# # Submit

# In[20]:


submission = pd.DataFrame({
    'id': test_id,
    'Target': submit_pred
})

# Save
submission.to_csv('submission.csv', index=False)

print(submission)


# In[ ]:




