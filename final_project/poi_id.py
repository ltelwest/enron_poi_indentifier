#!/usr/bin/python

import sys
import pickle
import numpy as np
from numpy import mean
import pandas as pd
from pandas import DataFrame
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

"""
Feature Selection & Grouping
The features in the dataset are split by email and financial features.
"""
features_email = ['to_messages', 'from_messages',  'from_poi_to_this_person',
           'from_this_person_to_poi', 'shared_receipt_with_poi']
# finance data
features_finance = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
             'deferral_payments', 'loan_advances', 'other', 'expenses',
             'director_fees', 'total_payments',
             'exercised_stock_options', 'restricted_stock',
             'restricted_stock_deferred', 'total_stock_value']
# all features
features_list = features_email + features_finance
# all features column names
features_column_names = ['poi'] + ['email_address'] + features_email + features_finance
# all features data type
features_dtype = [bool] + [str] + list(np.repeat(float, 19))

"""
Data Load
This identifier is the outcome of a ipython notebook which can be found in this
repo & folder (poi_id.ipynb).
The data will be loaded into a pandas dataframe for easier data wrangling and
compatability with the notebook.
"""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# converting the data into a data frame
df = DataFrame.from_dict(data_dict, orient='index')

# reordering the columns
df = df.loc[:, features_column_names]

# converting the data type
for i in xrange(len(features_column_names)):
    df[features_column_names[i]] = df[features_column_names[i]].astype(features_dtype[i], errors='ignore')

"""
Outlier removal
During the analysis in the notebook two wrong data points and misaligned data
for two employees were found which get fixed here.
"""
df = df[df.index != 'TOTAL']
df = df[df.index != 'THE TRAVEL AGENCY IN THE PARK']

df.loc['BELFER ROBERT', features_finance] = \
    [0, 0, 0, -102500, 0, 0, 0, 3285,
     102500, 3285, 0, 44093, -44093, 0]
df.loc['BHATNAGAR SANJAY', features_finance] = \
    [0, 0, 0, 0, 0, 0, 0, 137864, 0, 137864,
     15456290, 2604490, -2604490, 15456290]

"""
Feature creation
There are three measurable interactions with POIs in the dataset: sending,
recieving emails as well as sharing a reciept with a POI. Those interactions
should be translated into ratios that compare them with the total number of each
type of interaction.
"""

# calculate ratio
df['recieved_from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['sent_to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_receipt_with_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
# add labels to df
features_email_new = ['recieved_from_poi_ratio', 'sent_to_poi_ratio', 'shared_receipt_with_poi_ratio']
features_new = features_finance + features_email_new
features_basic = features_finance + features_email


"""
Feature scaling & Imputation
The features will be log scaled and missing values will be replaced with the
median. See notebook for further explanaition
"""
### Feature scaling
# log scaling
for f in features_list:
    df[f] = [np.log(abs(v)) if v != 0 else 0 for v in df[f]]

### Imputation
# replace null values
df.replace(0, np.NaN)
df.fillna(df.mean(), inplace=True)


"""
Get sklearn ready dataset
Transform dataframe to disctionary and define different sets of features.
"""
### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')


### Extract features and labels from dataset for local testing

#Select the labels
### Uncomment to select different feature sets and benchmark them
# basic
my_feature_list = ['poi'] + features_basic

# new
# my_feature_list = ['poi'] + features_new

# selected
# my_feature_list = ['poi'] + features_selected

# my_feature_list = ['poi'] + ['from_poi_to_this_person'] + ['shared_receipt_with_poi'] + ['restricted_stock'] + ['exercised_stock_options'] + ['bonus']
# print my_feature_list[1]


# Format and splot to labels and features
data = featureFormat(my_dataset, my_feature_list, remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)


"""
Algorithm selection
KMeans, SVC and RandomForestClassifier will be used & compared
"""

### selection of classifiers
# k_clf = KMeans()
# s_clf = SVC()
# rf_clf = RandomForestClassifier()


"""
Algorithm tuning
The evaluation methods of choice are accuracy, precision and recall.
The following function will return the values for each of those.
"""

### evaluation script using a 0.3 test split and evaluating based on precison & recall
# def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
#     print clf
#     accuracy = []
#     precision = []
#     recall = []
#     first = True
#     for trial in range(num_iters):
#         features_train, features_test, labels_train, labels_test =\
#             cross_validation.train_test_split(features, labels, test_size=test_size)
#         clf.fit(features_train, labels_train)
#         predictions = clf.predict(features_test)
#         accuracy.append(accuracy_score(labels_test, predictions))
#         precision.append(precision_score(labels_test, predictions))
#         recall.append(recall_score(labels_test, predictions))
#         if trial % 10 == 0:
#             if first:
#                 sys.stdout.write('\nProcessing')
#             sys.stdout.write('.')
#             sys.stdout.flush()
#             first = False
#
#     print "done.\n"
#     print "precision: {}".format(mean(precision))
#     print "recall:    {}".format(mean(recall))
#     return mean(precision), mean(recall)
#
# evaluate_clf(k_clf, features, labels)
# evaluate_clf(s_clf, features, labels)
# evaluate_clf(rf_clf, features, labels)

"""
Final Alogrithm Definition
SVC was identified as the best perfoming algorithm. After tweaking the parameters
systematically the following achieved precision & recall values above 0.3.
For further details see notebook.
"""

clf = SVC(kernel='rbf', C=2000,gamma = 0.0001,random_state = 42, class_weight = 'auto')

# For more details about other algorithms used please go to the algorithms
# section in poi_id.ipynb

test_classifier(clf,my_dataset, my_feature_list, folds = 200)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)
