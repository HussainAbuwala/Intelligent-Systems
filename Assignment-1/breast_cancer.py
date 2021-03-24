#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:21:07 2019

@author: hussainabuwala
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV


#read data from csv file
cancer_data = pd.read_csv('cancer.csv')

#drop unnamed column from the dataset. This column does not exist in original dataset.
cancer_data.drop(cancer_data.columns[cancer_data.columns.str.contains('unnamed', case=False)],
          axis=1, inplace=True)

#check if there is any null values in the dataser
cancer_data.isnull().sum()
cancer_data.isna().sum()

#looking at the datatypes of each column. Also looking at the descriptive statistics
cancer_data.dtypes
cancer_data.describe().to_csv("my_description.csv")

#seperating the features and the target class into two seperate dataframes
features = cancer_data.iloc[:,2:32].values
target_labels = cancer_data.iloc[:,1].values


#changing the target_class categorical values to numerical 0 and 1 values
labelencoder_target_labels = LabelEncoder()
target_labels = labelencoder_target_labels.fit_transform(target_labels)

#dividing the dataset into training and testing
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target_labels,
                                                                            test_size = 0.25,
                                                                            random_state = 0)


#Feature Scaling
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)



#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
classifier = KNeighborsClassifier(n_neighbors = 15,p = 2)
classifier.fit(features_train, target_train)
Y_pred = classifier.predict(features_test)
cm = confusion_matrix(target_test, Y_pred)
c = print((cm[0, 0] + cm[1, 1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]), "K-NEAREST-NEIGHBOURUR")
print(classification_report(target_test,Y_pred))
print(cm)
print("------------------------------------------------------------------------------------")


#Using SVC method of svm class to use Support Vector Machine Algorithm

classifier = SVC(kernel = 'linear', C = 0.1, gamma= 0.001)
classifier.fit(features_train, target_train)

Y_pred = classifier.predict(features_test)
cm = confusion_matrix(target_test, Y_pred)
c = print((cm[0, 0] + cm[1, 1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]), "SUPPORT VECTOR MACHINE")
print(classification_report(target_test,Y_pred))
print(cm)
print("------------------------------------------------------------------------------------")


#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 4,
                                    min_samples_split = 0.05, max_features = 15)
classifier.fit(features_train, target_train)

Y_pred = classifier.predict(features_test)
cm = confusion_matrix(target_test, Y_pred)
c = print((cm[0, 0] + cm[1, 1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]),"DECISION TREE")
print(classification_report(target_test,Y_pred))
print(cm)
print("------------------------------------------------------------------------------------")


#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
classifier = RandomForestClassifier(n_estimators = 10,max_depth = 5,
                                    max_features = 27,criterion = 'entropy', random_state = 0)
classifier.fit(features_train, target_train)

Y_pred = classifier.predict(features_test)
cm = confusion_matrix(target_test, Y_pred)
c = print((cm[0, 0] + cm[1, 1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]),"RANDOM FOREST")
print(classification_report(target_test,Y_pred))
print(cm)
print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")


'''

CODE FOR TUNING EACH ALGORITHM ARE GIVEN BELOW.

'''


#KNN PARAMETER TUNING

#NEIGHBORS PARAMETER TUNING
neighbors = list(range(1,30))
train_results = []
test_results = []
for n in neighbors:
    
   model = KNeighborsClassifier(n_neighbors=n)
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(neighbors, train_results, 'b', label="Train AUC")
line2, = plt.plot(neighbors, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_neighbors')
plt.show()


#DISTANCE METRIC TUNING
distances = [1, 2, 3, 4, 5]
train_results = []
test_results = []
for p in distances:
    
   model = KNeighborsClassifier(n_neighbors = 15,p=p)
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(distances, train_results, 'b', label="Train AUC")
line2, = plt.plot(distances, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('p')
plt.show()


print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")



#SVC PARAMTER TUNING

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
kernels = ['linear','rbf','poly']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernels }
grid_search = GridSearchCV(SVC(), param_grid)
grid_search.fit(features_train, target_train)
print('Best C:',grid_search.best_estimator_.C) 
print('Best Kernel:',grid_search.best_estimator_.kernel)
print('Best Gamma:',grid_search.best_estimator_.gamma)

print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")


#DECISION TREES PARAMETER TUNING


#DEPTH PARAMTER TUNING
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for d in max_depths:
    
   model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, 
                                  max_depth = d)
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max_depths')
plt.show()



#min_samples_split PARAMETER TUNING
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
test_results = []
for s in min_samples_splits:
    
   model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, 
                                  min_samples_split = s,max_depth = 4)
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min_samples_splits')
plt.show()


#max_features parameter tuning
max_features = list(range(1,features_train.shape[1]))
train_results = []
test_results = []
for f in max_features:
    
   model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, 
                                  min_samples_split = 0.05,max_depth = 4,
                                  max_features = f)
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max_features')
plt.show()

print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")





#RANDOM FOREST PARAMETER TUNING

#n_estimators parameter tuning
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
max_depths = np.linspace(1, 32, 32, endpoint=True)
max_features = list(range(1,features_train.shape[1]))

train_results = []
test_results = []
for n in n_estimators:
    
   model = RandomForestClassifier(n_estimators = n,
                                  criterion = 'entropy', random_state = 0)
   
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


#max_depths parameter tuning
train_results = []
test_results = []
for m in max_depths:
    
   model = RandomForestClassifier(n_estimators = 10,max_depth = m,
                                  criterion = 'entropy', random_state = 0)
   
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max_depths')
plt.show()



#max_features parameter tuning
train_results = []
test_results = []
for f in max_features:
    
   model = RandomForestClassifier(n_estimators = 10,max_depth = 5,max_features = f,
                                  criterion = 'entropy', random_state = 0)
   
   model.fit(features_train, target_train)
   train_pred = model.predict(features_train)
   
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   
   y_pred = model.predict(features_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
   
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max_features')
plt.show()
