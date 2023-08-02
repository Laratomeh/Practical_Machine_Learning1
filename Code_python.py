#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import nltk
import sklearn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from numpy import mean, absolute, sqrt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from nltk.corpus import wordnet as wn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score


# In[2]:


# Import datasets

class Dataset(object):

    def __init__(self):
        self.trainset = self.read_dataset("train_full.txt")
        self.testset = self.read_dataset("test.txt")

    def read_dataset(self, file_path):
        with open(file_path, encoding="UTF-8") as file:
            fieldnames = ['ID', 'sentence', 'start_index', 'end_index', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset


# In[3]:


# Print the length of the two datasets

data = Dataset()
print("{} training - {} test".format(len(data.trainset), len(data.testset)))


# In[4]:


# Features extraction

avg_word_length = 5.3
def extract_features(word):
        len_chars = len(word) / avg_word_length
        len_tokens = len(word.split(' '))
        pos = 0
        means = 0
        for wor in word.split(' '):
            s = [w.pos() for w in wn.synsets(word)]
            pos+=len(set(s))
            means+=len(s)
        #print([len_chars, len_tokens,pos/2,means/10])
        return [len_chars, len_tokens,pos/2,means/10]

train_X = []
train_y = []
dev_X = []
dev_y = []

for sent in data.trainset[:int(len(data.trainset)*0.8)+1]:
    train_X.append(extract_features(sent['target_word']))
    train_y.append(sent['gold_label'])

for sent in data.trainset[int(len(data.trainset)*0.8)+1:]:
    dev_X.append(extract_features(sent['target_word']))
    dev_y.append(sent['gold_label'])

test_X = []
test_Y = []

for sent in data.testset:
    test_X.append(extract_features(sent['target_word']))
    test_Y.append(sent['gold_label'])


# In[5]:


# Create Simple Linear Regression object
line_regr = LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=True)

# Train the model using the training sets
line_regr.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = line_regr.predict(dev_X)

# The coefficients
print("Coefficients: \n", line_regr.coef_)
# The mean squared error
print("Mean squared error: %.15f" % mean_squared_error(dev_y, pred_y))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.15f" % r2_score(dev_y, pred_y))
# The mean absolute error
print("Mean absolute error: %.15f" % mean_absolute_error(dev_y, pred_y))


# In[6]:


# Predict the test_Y values (The word complexity estimation of the test file)

test_Y = line_regr.predict(dev_X)
print(test_Y)


# In[7]:


plt.plot(test_Y)


# In[8]:


# Create data frame containes test_Y results 
datatest = pd.read_csv("test.csv")
Xtest = datatest.iloc[:, 0].values
new_data = [[Xtest[:], test_Y[:]]]
# Create the pandas DataFrame
df = pd.DataFrame(new_data, columns = [['ID','Label']])  
# print dataframe.
df


# In[9]:


# Save the results in the submission file

sub_file = pd.read_csv('submission.csv')
sub_file
label_result=pd.DataFrame(sub_file)
label_result['Label']=pd.Series(test_Y)
sub_file
label_result.to_csv('LinearRegression.csv')


# In[10]:


# Create Support Vector Regression object
svr_regr = SVR(kernel = 'rbf', epsilon = 0.0005, gamma=500.0)

# Train the model using the training sets
svr_regr.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = svr_regr.predict(dev_X)

# The mean squared error
print("Mean squared error: %.15f" % mean_squared_error(dev_y, pred_y))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.15f" % r2_score(dev_y, pred_y))
# The mean absolute error
print("Mean absolute error: %.15f" % mean_absolute_error(dev_y, pred_y))


# In[11]:


test_Y = svr_regr.predict(dev_X)
print(test_Y)


# In[12]:


plt.plot(test_Y)


# In[13]:


datatest = pd.read_csv("test.csv")
Xtest = datatest.iloc[:, 0].values
new_data = [[Xtest[:], test_Y[:]]] 
# Create the pandas DataFrame
df = pd.DataFrame(new_data, columns = [['ID','Label']])  
# print dataframe.
df


# In[14]:


# Save the results in the submission file

sub_file = pd.read_csv('submission.csv')
sub_file
label_result=pd.DataFrame(sub_file)
label_result['Label']=pd.Series(test_Y)
sub_file
label_result.to_csv('SupportVectorRegression.csv')


# In[15]:


# Creat Decision Tree Regression object
dtr_regr = DecisionTreeRegressor(ccp_alpha=0.000001, random_state=100,
                                 min_samples_leaf=40,min_samples_split=10,
                                 splitter='best' 
                                )

# Train the model using the training sets
dtr_regr.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = dtr_regr.predict(dev_X)

# The mean squared error
print("Mean squared error: %.15f" % mean_squared_error(dev_y, pred_y))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.15f" % r2_score(dev_y, pred_y))
# The mean absolute error
print("Mean absolute error: %.15f" % mean_absolute_error(dev_y, pred_y))


# In[16]:


test_Y = dtr_regr.predict(dev_X)
print(test_Y)


# In[17]:


plt.plot(test_Y)


# In[18]:


datatest = pd.read_csv("test.csv")
Xtest = datatest.iloc[:, 0].values
new_data = [[Xtest[:], test_Y[:]]] 
# Create the pandas DataFrame
df = pd.DataFrame(new_data, columns = [['ID','Label']])  
# print dataframe.
df


# In[19]:


# Save the results in the submission file

sub_file = pd.read_csv('submission.csv')
sub_file
label_result=pd.DataFrame(sub_file)
label_result['Label']=pd.Series(test_Y)
sub_file
label_result.to_csv('DecisionTreeRegression.csv')


# In[20]:


# Create Random Forest Regression model object
rf_regr = RandomForestRegressor(n_estimators = 7000, random_state = 1000)

# Train the model using the training sets
rf_regr.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = rf_regr.predict(dev_X)

# The mean squared error
print("Mean squared error: %.15f" % mean_squared_error(dev_y, pred_y))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.15f" % r2_score(dev_y, pred_y))
# The mean absolute error
print("Mean absolute error: %.15f" % mean_absolute_error(dev_y, pred_y))


# In[21]:


test_Y = rf_regr.predict(dev_X)
print(test_Y)


# In[22]:


plt.plot(test_Y)


# In[23]:


datatest = pd.read_csv("test.csv")
Xtest = datatest.iloc[:, 0].values
new_data = [[Xtest[:], test_Y[:]]] 
# Create the pandas DataFrame
df = pd.DataFrame(new_data, columns = [['ID','Label']])  
# print dataframe.
df


# In[24]:


# Save the results in the submission file

sub_file = pd.read_csv('submission.csv')
sub_file
label_result=pd.DataFrame(sub_file)
label_result['Label']=pd.Series(test_Y)
sub_file
label_result.to_csv('RandomForestRegression.csv')


# In[25]:


# Define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Use k-fold CV to evaluate model
scores = cross_val_score(line_regr, train_X, train_y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# View mean absolute error (MAE)
print("Mean Absolute Error is :  ", mean(absolute(scores)))

# View root mean squared error (RMSE)
# print("Root Mean Squared Error is :  ", sqrt(mean(absolute(scores))))


# In[26]:


# Define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Use k-fold CV to evaluate model
scores = cross_val_score(svr_regr, train_X, train_y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# View mean absolute error (MAE)
print("Mean Absolute Error is :  ", mean(absolute(scores)))

# View root mean squared error (RMSE)
# print("Root Mean Squared Error is :  ", sqrt(mean(absolute(scores))))


# In[27]:


# Define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Use k-fold CV to evaluate model
scores = cross_val_score(dtr_regr, train_X, train_y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# View mean absolute error (MAE)
print("Mean Absolute Error is :  ", mean(absolute(scores)))

# View root mean squared error (RMSE)
# print("Root Mean Squared Error is :  ", sqrt(mean(absolute(scores))))


# In[28]:


# Define cross-validation method to use
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# Use k-fold CV to evaluate model
scores = cross_val_score(rf_regr, train_X, train_y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# View mean absolute error (MAE)
print("Mean Absolute Error is :  ", mean(absolute(scores)))

# View root mean squared error (RMSE)
# print("Root Mean Squared Error is :  ", sqrt(mean(absolute(scores))))

