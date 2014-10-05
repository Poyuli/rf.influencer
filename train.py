import os
import csv
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import svm, grid_search
from sklearn.cross_validation import cross_val_score

os.chdir("/Users/BradLi/Documents/Data Science/Kaggle/Influencer")
writeFile = False
model = "SVM"

# Merging training and test sets
df_train = pd.read_csv("train.csv", header = 0)
df_test = pd.read_csv("test.csv", header = 0)
df = df_train.append(df_test, ignore_index = True)
label = df['Choice']
df = df.iloc[0::,0:-1].replace(0,0.1)
df['Choice'] = label
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# Feature engineering
df['follower'] = df.A_follower_count.apply(math.log) - df.B_follower_count.apply(math.log)
df['following'] = df.A_following_count.apply(math.log) - df.B_following_count.apply(math.log)
df['listed'] = df.A_listed_count.apply(math.log) - df.B_listed_count.apply(math.log)
df['mrcv'] = df.A_mentions_received.apply(math.log) - df.B_mentions_received.apply(math.log)
df['msnt'] = df.A_mentions_sent.apply(math.log) - df.B_mentions_sent.apply(math.log)
df['posts'] = df.A_posts.apply(math.log) - df.B_posts.apply(math.log)
df['rrcv'] = df.A_retweets_received.apply(math.log) - df.B_retweets_received.apply(math.log)
df['rsnt'] = df.A_retweets_sent.apply(math.log) - df.B_retweets_sent.apply(math.log)
df['ft1'] = df.A_network_feature_1.apply(math.log) - df.B_network_feature_1.apply(math.log)
df['ft2'] = df.A_network_feature_2.apply(math.log) - df.B_network_feature_2.apply(math.log)
df['ft3'] = df.A_network_feature_3.apply(math.log) - df.B_network_feature_3.apply(math.log)
df = df.drop(['A_follower_count','B_follower_count','A_following_count','B_following_count','A_listed_count','B_listed_count'], axis=1)
df = df.drop(['A_mentions_received','B_mentions_received','A_mentions_sent','B_mentions_sent','A_posts','B_posts'], axis=1)
df = df.drop(['A_retweets_received','B_retweets_received','A_retweets_sent','B_retweets_sent'], axis=1)
df = df.drop(['A_network_feature_1','B_network_feature_1','A_network_feature_2','B_network_feature_2','A_network_feature_3','B_network_feature_3'], axis=1)

df['engage'] = df.msnt + df.rsnt + df.posts
df['ratio_f'] = df.follower - df.following
df['ratio_r'] = df.rrcv - df.rsnt
df['ratio_m'] = df.mrcv - df.msnt
df['ft23'] = df.ft2 + df.ft3
df = df.drop(['msnt','rsnt','posts','mrcv','rrcv','following','ft2','ft3'], axis=1)

if model == "SVM":
    df = (df - df.min()) / (df.max() - df.min())

data = df.values

# Random forest or SVM training
if model == "RF":
    forest = RFC(n_estimators = 100)
    cv_score = cross_val_score(forest, data[0:5500,1::], data[0:5500,0], cv=10)
    print "CV Score = ", cv_score.mean(),"\n"
    forest = forest.fit(data[0:5500,1::], data[0:5500,0])
    output = forest.predict_proba(data[5500::,1::])
    features = df.columns.tolist()[1:]
    print "Feature importances:"
    print zip(features, forest.feature_importances_)
    
elif model == "SVM":
    svc = svm.SVC()
    param = {'C':[1e3,1e2,1e1,1e0,1e-1,1e-2], 'gamma':[1e-1,1e0,1e1,1e2,1e3,1e4], 'kernel':['rbf']}
    svc = grid_search.GridSearchCV(svc, param, cv=10)
    svc.fit(data[0:5500,1::], data[0:5500,0])
    output = svc.predict(data[5500::,1::])
    print "Optimized parameters:"
    print svc.best_estimator_
    print "Best CV Score = "
    print svc.best_score_

# Writing predicted results to csv file
if writeFile == True:
    pfile = open("submission.csv","w+")
    p = csv.writer(pfile)
    p.writerow(['Id','Choice'])
    idx = 1
    for item in output:
        p.writerow([idx,item[0]])
        idx += 1    
    pfile.close()

