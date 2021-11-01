# real predictions done here
from numpy.random import normal
import pandas as pd
import numpy as np
import os
import sys
from math import comb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from functions import *

# change working directory to file location
# os.chdir('src/')


# load data
X_train = pd.read_csv("../data/X_train.csv").drop(['id'], axis=1)
X_test = pd.read_csv("../data/X_test.csv").drop(['id'], axis=1)
y_train = pd.read_csv("../data/y_train.csv").drop(['id'], axis=1)


# code such that test and train data are both manipulated at the same time
X = pd.concat([X_train, X_test])
# impute data with median value 
X = X.fillna(X.median())
X = X.drop(X.loc[:, X.var()<0.0001].columns, axis=1)
X = normalise(X) # normalise the data
X['data'] = ['train'] * X_train.shape[0]  +  ['test'] * X_test.shape[0]
X_train = X.loc[X['data'] == 'train'].drop("data", axis=1)
X_test = X.loc[X['data'] == 'test',:].drop("data", axis=1)


# select the variables that seem to matter the most to reduce dimensionality
# run Lasso to see which alpha gives the highest crossvalidation score
lasso_baseline = LassoCV(cv=10, random_state=42, tol=1e-2).fit(X_train, np.ravel(y_train))
# then run new Lasso with slightly lower alpha to have a few features added that could be relevant
alpha = lasso_baseline.alpha_ - 0.05
lasso_select = Lasso(alpha=alpha).fit(X_train, np.ravel(y_train))
selected_features = pd.DataFrame({'variable' : X_train.columns[abs(lasso_select.coef_) > 0.0001], 'coef' : lasso_select.coef_[abs(lasso_select.coef_) > 0.0001]}).sort_values(by='coef', ascending=False)
X_train_selected = X_train[selected_features['variable'].values]


# do feature engineering on training set
model = LinearRegression()
path = '../out/X_train_engineered.csv'
X_train_engineered = feature_engineering(X_train_selected, y_train, model, 5, 4, 'r2', 5, 50, path)


# apply engineering to test data
features = X_train_engineered.columns
path = '../out/X_test_engineered.csv'
X_test_engineered = engineered_testdata(X_test, features, path)


# fit prediction model
X_train_engineered = pd.read_csv('../out/X_train_engineered.csv')
X_train_engineered = normalise(X_train_engineered)
lasso_predict = LassoCV(cv=10, random_state=42, ).fit(X_train_engineered, np.ravel(y_train))
score = lasso_predict.score(X_train_engineered, np.ravel(y_train))
print('The crossvalidation score is: ', score, " while the baseline score is: ", lasso_baseline.score(X_train, np.ravel(y_train)))


# predict based on engineered test data
X_test_engineered = pd.read_csv('../out/X_test_engineered.csv').iloc[:,1:]
X_test_engineered = normalise(X_test_engineered)
y_test = pd.DataFrame({'id' : range(X_test_engineered.shape[0])})
y_test['y'] = lasso_predict.predict(X_test_engineered)
y_test.to_csv('../out/y_test.csv', index=False)