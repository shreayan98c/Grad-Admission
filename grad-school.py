#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


# Importing the dataset
data = pd.read_csv('dataset/data.csv')
data.head()


# In[3]:


data.dtypes


# ## Inspecting the data

# In[4]:


data.describe()


# In[5]:


data.isna().sum()


# In[6]:


data = data.drop('Serial No.', axis=1)


# In[7]:


data.head()


# ## Data Visualizations

# In[8]:


# Heatmap
corr = data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':20}, cmap='Greens')
plt.title('Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


# In[9]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(2,2, figsize=(20,12))               
plt.suptitle("Distribution of GRE scores", fontsize=20)
sns.countplot(data['GRE Score'], ax = ax[0,0])
sns.distplot(data['GRE Score'], ax = ax[1,0]) 
sns.violinplot(data['GRE Score'], ax = ax[0,1])
sns.boxplot(data['GRE Score'], ax = ax[1,1]) 
plt.show()


# In[10]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(2,2, figsize=(20,12))               
plt.suptitle("Distribution of TOEFL scores", fontsize=20)
sns.countplot(data['TOEFL Score'], ax = ax[0,0])
sns.distplot(data['TOEFL Score'], ax = ax[1,0]) 
sns.violinplot(data['TOEFL Score'], ax = ax[0,1])
sns.boxplot(data['TOEFL Score'], ax = ax[1,1]) 
plt.show()


# In[11]:


sns.countplot(data['University Rating'])
plt.show()


# In[12]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(2,2, figsize=(20,12))               
plt.suptitle("Distribution of SOP scores", fontsize=20)
sns.countplot(data['SOP'], ax = ax[0,0])
sns.distplot(data['SOP'], ax = ax[1,0]) 
sns.violinplot(data['SOP'], ax = ax[0,1])
sns.boxplot(data['SOP'], ax = ax[1,1]) 
plt.show()


# In[13]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(2,2, figsize=(20,12))               
plt.suptitle("Distribution of LOR scores", fontsize=20)
sns.countplot(data['LOR '], ax = ax[0,0])
sns.distplot(data['LOR '], ax = ax[1,0]) 
sns.violinplot(data['LOR '], ax = ax[0,1])
sns.boxplot(data['LOR '], ax = ax[1,1]) 
plt.show()


# In[14]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(2,2, figsize=(20,12))               
plt.suptitle("Distribution of CGPA", fontsize=20)
sns.countplot(data['CGPA'], ax = ax[0,0])
sns.distplot(data['CGPA'], ax = ax[1,0]) 
sns.violinplot(data['CGPA'], ax = ax[0,1])
sns.boxplot(data['CGPA'], ax = ax[1,1]) 
plt.show()


# In[15]:


sns.countplot(data['Research'])
plt.show()


# In[16]:


sns.distplot(data['Chance of Admit '])
plt.show()


# ## Building the Models

# In[17]:


# Supervised Learning
X = data.drop('Chance of Admit ', axis=1)
y = data['Chance of Admit ']


# In[18]:


# Splitting the data into training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)


# In[19]:


# # Normalization
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# ## Linear Regression

# In[20]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[21]:


# Predicting the Test set results
y_pred = lr.predict(X_test)


# In[22]:


# Calculating the r-squared score
from sklearn.metrics import r2_score
acc_lr = round(r2_score(y_test, y_pred) * 100, 2)
print("R-squared Score : ", acc_lr)


# ## Random Forest Regression

# In[23]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(X_train,y_train)


# In[24]:


# Predicting the Test set results
y_pred = rfr.predict(X_test)


# In[25]:


# Calculating the r-squared score
from sklearn.metrics import r2_score
acc_rfr = round(r2_score(y_test, y_pred) * 100, 2)
print("R-squared Score : ", acc_rfr)


# ## Decision Tree Regression

# In[26]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train,y_train)


# In[27]:


# Predicting the Test set results
y_pred = dtr.predict(X_test)


# In[28]:


# Calculating the r-squared score
from sklearn.metrics import r2_score
acc_dtr = round(r2_score(y_test, y_pred) * 100, 2)
print("R-squared Score : ", acc_dtr)


# ## SVM Regressor

# In[29]:


from sklearn import svm
svm = svm.SVR()
svm.fit(X_train, y_train)


# In[30]:


# Predicting the Test set results
y_pred = svm.predict(X_test)


# In[31]:


# Calculating the r-squared score
from sklearn.metrics import r2_score
acc_svm = round(r2_score(y_test, y_pred) * 100, 2)
print("R-squared Score : ", acc_svm)


# ## XGBoost Regressor

# In[32]:


from xgboost import XGBRegressor
gbm = XGBRegressor()
gbm.fit(X_train, y_train)


# In[33]:


# Predicting the Test set results
y_pred = gbm.predict(X_test)


# In[34]:


# Calculating the r-squared score
from sklearn.metrics import r2_score
acc_xgb = round(r2_score(y_test, y_pred) * 100, 2)
print("R-squared Score : ", acc_xgb)


# ## Evaluating and Comparing the Models

# In[35]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'SVM Regressor', 'XGBoost'],
    'R-squared Score': [acc_lr, acc_dtr, acc_rfr, acc_svm, acc_xgb]})
models.sort_values(by='R-squared Score', ascending=False)


# In[36]:


# # Save the best model as a pickle string
saved_model = pickle.dumps(lr)


# In[37]:


# Store data (serialize)
with open('model.pickle', 'wb') as file:
    pickle.dump(lr, file)


# In[38]:


# Save the best model as a pickle in a file 
# joblib.dump(lr, 'model.pkl')


# ### Hence Linear Regression gives the best performance in this scenario

# In[39]:


# Predicting on a sample dataframe
sample = pd.DataFrame([[335, 116, 5, 5, 5, 9.7, 1]], columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
sample.head()


# In[40]:


# # Predicting the Test set results
# sample_pred = lr.predict(sample)
# sample_pred


# In[41]:


# # Load the pickled model 
lr_from_pickle = pickle.loads(saved_model) 


# In[42]:


# Load data (deserialize)
with open('model.pickle', 'rb') as file:
    lr_from_pickle = pickle.load(file)


# In[43]:


# Load the pickled model from the file 
# model_from_joblib = joblib.load('model.pkl')


# In[44]:


# Predicting the sample result
sample_pred = lr_from_pickle.predict(sample)
sample_pred


# ### Recommending universities based on profile and scores
# The University recommendations will be based on the World's Top Universities ranking from this website: <br>
# https://www.topuniversities.com/student-info/choosing-university/worlds-top-100-universities

# In[45]:


if(sample_pred[0] > 0.95):
    univ_list = ['Harvard', 'Stanford', 'MIT']


# ### The output score will be a value between 0&1. <br>TODO: We need to take care of perfect or fake values.
