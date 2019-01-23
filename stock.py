# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:47:27 2019

@author: yaoyitong
"""

import pandas as pd

msft_data_copy=pd.read_csv('D:/BaiduYunDownload\msft_stockprices_dataset.csv',encoding="utf-8")

msft_data_copy["diff_low_high"]=msft_data_copy["High Price"]-msft_data_copy["Low Price"]
#msft_data_copy["diff_low_open"]=msft_data_copy["Open Price"]-msft_data_copy["Low Price"]
#msft_data_copy["diff_open_high"]=msft_data_copy["High Price"]-msft_data_copy["Open Price"]

msft_data_copy_x=msft_data_copy.drop(["Date","Close Price"],axis=1)
msft_data_copy_y=msft_data_copy["Close Price"]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(msft_data_copy_x),columns=msft_data_copy_x.columns)

train_x=data_minmax
Y=msft_data_copy_y

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
KF=KFold(n_splits=5)
from sklearn import ensemble
import xgboost as xgb

params={'booster':'gbtree',
	    'objective': 'reg:linear',
	    'eval_metric':'rmse',
	    'gamma':0.6,
	    'min_child_weight':3,
	    'max_depth':6,
       'n_estimators':1900,
	    'lambda':10,
	    'subsample':0.6,
	    'colsample_bytree':0.6,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12,
        'learning_rate':0.01,
        'silent':1,
        'verbose_eval':500
	    }


obs_cv_traindata_y=pd.DataFrame()
rf_cv_traindata_y=pd.DataFrame()
gbdt_cv_traindata_y=pd.DataFrame()
xgb_cv_traindata_y=pd.DataFrame()

for train_index,test_index in KF.split(train_x): 
    
    cv_traindata_x=train_x.loc[train_index]
    cv_traindata_y=Y.loc[train_index]
    
    cv_testdata_x=train_x.loc[test_index]
    cv_testdata_y=Y.loc[test_index]
    obs_cv_traindata_y=obs_cv_traindata_y.append(pd.DataFrame(cv_testdata_y))
    

    rf0 = RandomForestRegressor(oob_score=True, random_state=10,
      n_estimators=600,max_depth=10,min_samples_split=6)

    rf0.fit(cv_traindata_x,cv_traindata_y)
    cv_testdata_rf=rf0.predict(cv_testdata_x)
    cv_testdata_rf=pd.DataFrame(cv_testdata_rf)
    cv_testdata_rf.columns=["Close Price"]
    rf_cv_traindata_y=rf_cv_traindata_y.append(pd.DataFrame(cv_testdata_rf))
    
    clf = ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=3,
      min_samples_leaf=7, learning_rate=0.12,loss='ls')
    gbdt_model = clf.fit(cv_traindata_x,cv_traindata_y)  # Training model
    cv_testdata_gbdt = gbdt_model.predict(cv_testdata_x)
    cv_testdata_gbdt=pd.DataFrame(cv_testdata_gbdt)
    cv_testdata_gbdt.columns=["Close Price"]
    gbdt_cv_traindata_y=gbdt_cv_traindata_y.append(pd.DataFrame(cv_testdata_gbdt))

    dataset1 = xgb.DMatrix(cv_traindata_x,cv_traindata_y)
    dataset2 = xgb.DMatrix(cv_testdata_x)
    watchlist = [(dataset1,'train')]
    model = xgb.train(params,dataset1,num_boost_round=1900,evals=watchlist)
    cv_testdata_xgb=model.predict(dataset2)
    cv_testdata_xgb=pd.DataFrame(cv_testdata_xgb)
    cv_testdata_xgb.columns=["Close Price"]
    xgb_cv_traindata_y=xgb_cv_traindata_y.append(pd.DataFrame(cv_testdata_xgb))



rf_cv_traindata_y.reset_index(drop=True, inplace=True)
gbdt_cv_traindata_y.reset_index(drop=True, inplace=True)
xgb_cv_traindata_y.reset_index(drop=True, inplace=True)

### random Forest
error_perc=(rf_cv_traindata_y-obs_cv_traindata_y)/obs_cv_traindata_y
error_tolerance=0.1  # 0.05
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("Ranfom Forest \n误差率为10%时 股票价格预测准确率: {:.2f}%".format(precision[0]))

error_tolerance=0.05 # 0.1
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("Ranfom Forest \n误差率为5%时 股票价格预测准确率: {:.2f}%".format(precision[0]))


### GBDT
error_perc=(gbdt_cv_traindata_y-obs_cv_traindata_y)/obs_cv_traindata_y
error_tolerance=0.1  # 0.05
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("GBDT \n误差率为10%时 股票价格预测准确率: {:.2f}%".format(precision[0]))

error_tolerance=0.05 # 0.1
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("GBDT \n误差率为5%时 股票价格预测准确率: {:.2f}%".format(precision[0]))


### XGBoost
error_perc=(xgb_cv_traindata_y-obs_cv_traindata_y)/obs_cv_traindata_y
error_tolerance=0.1  # 0.05
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("XGBoost \n误差率为10%时 股票价格预测准确率: {:.2f}%".format(precision[0]))

error_tolerance=0.05 # 0.1
precision=(error_perc<error_tolerance).sum(axis=0)/len(obs_cv_traindata_y)*100
print("XGBoost \n误差率为5%时 股票价格预测准确率: {:.2f}%".format(precision[0]))

