# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:15:23 2017

@author: VIGNESH
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:18:40 2017

@author: VIGNESHWAR
"""

import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#data preprocessing
sub_ids = test['transaction_id']
test = test.drop('transaction_id', axis=1)
y = train['target']
train = train.drop(['transaction_id', 'target'], axis=1)

#removing the dataset which has same values and the unwanted columns
train = train.drop(['num_var_3', 'cat_var_1','cat_var_3','cat_var_6','cat_var_8','cat_var_13','cat_var_27','cat_var_28','cat_var_31','cat_var_32','cat_var_33','cat_var_35','cat_var_36','cat_var_37','cat_var_38','cat_var_39','cat_var_41','cat_var_42'], axis=1)
test = test.drop(['num_var_3', 'cat_var_1','cat_var_3','cat_var_6','cat_var_8','cat_var_13','cat_var_27','cat_var_28','cat_var_31','cat_var_32','cat_var_33','cat_var_35','cat_var_36','cat_var_37','cat_var_38','cat_var_39','cat_var_41','cat_var_42'], axis=1)


#enocoding 
from sklearn.preprocessing import LabelEncoder
cat_vars = [x for x in train.columns if 'cat_' in x]
len(cat_vars)
for x in cat_vars:
    train[x] = train[x].fillna('NaN')
    test[x] = test[x].fillna('NaN')
    encoder = LabelEncoder()
    encoder.fit(list(set(list(train[x]) + list(test[x]))))
    train[x] = encoder.transform(train[x])
    test[x] = encoder.transform(test[x])

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
train = sc_x.fit_transform(train)
test = sc_x.fit_transform(test)
   

#forestclassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

forest_clf = RandomForestClassifier(random_state=7)

y_probas_forest = cross_val_predict(forest_clf, train, y, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]

roc_auc_score(y, y_scores_forest)


# fit on the whole training set
forest_clf = RandomForestClassifier(random_state=7)
forest_clf.fit(train, y)


preds = forest_clf.predict_proba(test)[:,1]



from IPython.display import FileLink

sub = pd.DataFrame({'transaction_id': sub_ids, 'target': preds})
sub = sub[['transaction_id','target']]    

filename='solution.csv'
sub.to_csv(filename, index=False)
FileLink(filename) 