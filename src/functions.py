from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from math import comb

def normalise(X):
    X = (X-X.mean())/X.std()
    return  X

def sort_feature_names(s, orig_features):
    # sorts s based on the order in orig_features which makes sure that features that combine the same columns have the same name and can be dropped base on the name
    index = [(lambda x: orig_features.index(x))(x) for x in s.split(':')]
    zipped_lists = zip(index, s.split(':'))
    sorted_zipped_lists = sorted(zipped_lists)
    
    return ":".join([element for _, element in sorted_zipped_lists])


def feature_engineering(X, y, model, folds, deg_poly, score, n_jobs, top, path):

    crossvalidation = KFold(n_splits=folds, shuffle=True, random_state=42)
    orig_features = list(X.columns.drop('interaction', errors='ignore'))
    tested_interactions = []

    for d in range(deg_poly): # loop over the degree of polynomials
        features = list(X.columns.drop('interaction', errors='ignore'))

        # only interact the current features with the new features to avoid working twice; could be deleted
        if 'new_features' in locals(): 
            features_interact = new_features 
            num_poly = comb(len(features_interact),2) + len(features_interact)*len(features)
        else: 
            features_interact = orig_features
            num_poly = comb(len(features),2) + len(features)

        baseline = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))
        data_interactions = pd.DataFrame(index=range(X.shape[0]) , columns=range(num_poly)) # stores all interactions
        eval_interactions = pd.DataFrame(columns=['feature_A:feature_B', score], index=range(num_poly)) #  stores how good the interaction was

        i=0    
        for feature_A in features_interact: # features that are interacted with current features
            for feature_B in features:
                
                name_interaction = sort_feature_names(feature_A + ":" + feature_B, orig_features)
                
                if name_interaction not in tested_interactions: # features.index(feature_A) >= features.index(feature_B): # >= to create x^2 etc. (but x^3 is only created if x^2 has been chosen) # make sure this interaction has not been done
                    tested_interactions.append(name_interaction) # save that this interaction has been calculated

                    X['interaction'] = X[feature_A] * X[feature_B]
                    score_eval = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))
                    
                    if score_eval > baseline: # only store new interaction if it improves on baseline
                        eval_interactions.iloc[i, : ] = pd.Series({'feature_A:feature_B' : name_interaction, score : round(score_eval,4)})
                        data_interactions.iloc[:, i] = X['interaction'] # store the good interaction data
                        data_interactions.rename(columns={i : name_interaction}, inplace=True) # give column the feature name
                        i+=1

        # check "pure" polynomials
        for feature in orig_features:

            name_interaction = ":".join([feature]*d)
            
            if name_interaction not in tested_interactions: # features.index(feature_A) >= features.index(feature_B): # >= to create x^2 etc. (but x^3 is only created if x^2 has been chosen) # make sure this interaction has not been done
                tested_interactions.append(name_interaction) # save that this interaction has been calculated

                X['interaction'] = np.power(X[feature], d)
                score_eval = np.mean(cross_val_score(model, X, y, scoring=score, cv=crossvalidation, n_jobs=n_jobs))

                if score_eval > baseline: # only store new interaction if it improves on baseline
                            eval_interactions.iloc[i, : ] = pd.Series({'feature_A:feature_B' : name_interaction, score : round(score_eval,4)})
                            data_interactions.iloc[:, i] = X['interaction'] # store the good interaction data
                            data_interactions.rename(columns={i : name_interaction}, inplace=True) # give column the feature name
                            i+=1

        # choose features and define X new
        new_features = eval_interactions.sort_values(by=score, ascending=False, ignore_index=True).loc[0:top, 'feature_A:feature_B'] # new features are the top ones of the engineered ones
        X = pd.concat([X.drop('interaction', axis=1, errors='ignore'), data_interactions[new_features]], axis=1)

    X.to_csv(path, index=False)

    return X


def engineered_testdata(X_test, features, path):
    # takes the test data and manipulates it in the same way the training data was manipulated
    n =  X_test.shape[0]
    X_test_eng = pd.DataFrame(columns=features, index=range(n))
    for feature in features:
        cols = feature.split(':')
        feat_eng = np.ones((1,n))
        for col in cols:
            feat_eng = feat_eng * np.array(X_test[col])
        X_test_eng[feature] = feat_eng.T

    X_test_eng.to_csv(path, index=False)

    return X_test_eng