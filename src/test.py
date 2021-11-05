import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.abod import ABOD
from sklearn.ensemble import IsolationForest

# compare two outlier methods: do some crude missing variable imputation
df_x = pd.read_csv (r'C:\Users\Felix\Dropbox\Courses\Year 2\Advanced Machine Learning\task1\X_train.csv')
X = df_x.fillna(df_x.mean())
X = X.drop(X.loc[:, X.var()<0.0001].columns, axis=1).to_numpy()
X = StandardScaler().fit_transform(X)


#Isolation forest (super fast)
isoF_model = IsolationForest(n_estimators = 2000, random_state=0, contamination = 0.05)
isoF = isoF_model.fit_predict(X) == -1 # can use later to retrieve y
isoF_inliers = np.squeeze(X[np.where(isoF == 0), :])
pd.DataFrame(isoF_inliers).to_csv('../out/iso_X.csv', index=False)
#isoF computes a score that can be used to correlate with abod score
isoF_scores = isoF_model.score_samples(X)

#Angle Based outlier detection
ab = ABOD(method = "default", contamination = 0.05)
ab.fit(X)
ab_index = ab.labels # can also use later to retrieve y
abod_inliers = np.squeeze(X[np.where(ab.labels_ == 0), :])
pd.DataFrame(abod_inliers).to_csv('../out/abod_X.csv', index=False)
abod_scores = ab.decision_scores_ 