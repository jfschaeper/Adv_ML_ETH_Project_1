# real predictions done here
import os
# change working directory to file location
# os.chdir('src/')

from numpy.random import normal
import pandas as pd
import numpy as np
import sys
from math import comb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from functions import *
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # needed for MissForest
from missingpy import MissForest
import time 
from sklearn.ensemble import IsolationForest
from scipy.stats.mstats import winsorize

# load data
X_train = pd.read_csv("../data/X_train.csv").drop(['id'], axis=1)
X_test = pd.read_csv("../data/X_test.csv").drop(['id'], axis=1)
y_train = pd.read_csv("../data/y_train.csv").drop(['id'], axis=1)


# code such that test and train data are both manipulated at the same time
X = pd.concat([X_train, X_test])
X = X.drop(X.loc[:, X.var()<0.0001].columns, axis=1)

# Impute 1:
# impute data with median value 
# X_imputed = X.fillna(X.median())

# Impute 2:
# print("Imputing with MissForest. Will take a long time")
# imputer = MissForest(max_iter=5, n_estimators=30, criterion='squared_error', max_features=None)
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# X_imputed.to_csv('../out/X_imputed.csv', index=False)

# outlier:
X_imputed = pd.read_csv('../out/X_imputed.csv')
#Isolation forest (super fast)
# isoF_model = IsolationForest(n_estimators = 2000, random_state=0, contamination = 0.05)
# isoF = isoF_model.fit(X_imputed) # can use later to retrieve y
# isoF.predict(X_imputed.x1)
# isoF_inliers = np.squeeze(X_imputed[np.where(isoF == 1), :])
# pd.DataFrame(isoF_inliers).to_csv('../out/iso_X.csv', index=False)
# #isoF computes a score that can be used to correlate with abod score
# isoF_scores = isoF_model.score_samples(X_imputed)

# outlier based on z score per column and cap values at +-2 standard deviations for each column
# X_no_outlier = correct_outlier(X_imputed, 2)
# winsorize
X_no_outlier = X_imputed.apply(lambda x: winsorize(x ,limits=[0.05, 0.05]), axis=0)
# save data
X_no_outlier.to_csv('../out/X_no_outlier.csv', index=False)

# split test and train again
X_no_outlier = pd.read_csv("../out/X_no_outlier.csv")
X = normalise(X_no_outlier) # normalise the data
X['data'] = ['train'] * X_train.shape[0]  +  ['test'] * X_test.shape[0]
X_train = X.loc[X['data'] == 'train'].drop("data", axis=1)
X_test = X.loc[X['data'] == 'test',:].drop("data", axis=1)

# select the variables that seem to matter the most to reduce dimensionality
# run Lasso to see which alpha gives the highest crossvalidation score
lasso_baseline = LassoCV(cv=10, random_state=42, tol=1e-2).fit(X_train, np.ravel(y_train))
# then run new Lasso with slightly lower alpha to have a few features added that could be relevant
alpha = lasso_baseline.alpha_ - 0.01
lasso_select = Lasso(alpha=alpha).fit(X_train, np.ravel(y_train))
selected_features = pd.DataFrame({'variable' : X_train.columns[abs(lasso_select.coef_) > 0.0001], 'coef' : lasso_select.coef_[abs(lasso_select.coef_) > 0.0001]}).sort_values(by='coef', ascending=False)
X_train_selected = X_train[selected_features['variable'].values]


# do feature engineering on training set
print("feature engineering")
model = LinearRegression()
path = '../out/X_train_engineered.csv'
X_train_engineered = feature_engineering(X_train_selected, y_train, model, 7, 3, 'r2', -1, 50, path)


# apply engineering to test data
features = X_train_engineered.columns
path = '../out/X_test_engineered.csv'
X_test_engineered = engineered_testdata(X_test, features, path)


# fit prediction model
X_train_engineered = pd.read_csv('../out/X_train_engineered.csv')
X_train_engineered = normalise(X_train_engineered)
lasso_predict = LassoCV(cv=10, random_state=42).fit(X_train_engineered, np.ravel(y_train))
score = lasso_predict.score(X_train_engineered, np.ravel(y_train))
print('The crossvalidation score is: ', score, " while the baseline score is: ", lasso_baseline.score(X_train, np.ravel(y_train)))


# predict based on engineered test data
X_test_engineered = pd.read_csv('../out/X_test_engineered.csv')
X_test_engineered = normalise(X_test_engineered)
y_test = pd.DataFrame({'id' : range(X_test_engineered.shape[0])})
y_test['y'] = lasso_predict.predict(X_test_engineered)
y_test.to_csv('../out/y_test.csv', index=False)