
# coding: utf-8

# # Case Study: Breast Cancer Classification
# 
# ### K. Sai Mallesh
# 
# ## Step-1: Problem Statement

# -  Predictingk if the cancer diagnosis is **benign** or **malignant** based on several observations/features.
# 
# -  30 features are used, examples:
# 
#     -  radius (mean of distances from center to points on the perimeter)
# 
#     -  texture (standard deviation of gray-scale values)
# 
#     -  perimeter
# 
#     -  area
# 
#     -  smoothness (local variation in radius lengths)
# 
#     -  compactness (perimeter^2 / area - 1.0)
# 
#     -  concavity (severity of concave portions of the contour)
# 
#     -  concave points (number of concave portions of the contour)
# 
#     -  symmetry
# 
#     -  fractal dimension ("coastline approximation" - 1)
#   
# -  Number of instances: 569
# 
# -  Number of attributes: 32(ID, diagnosis, 30 real-valued input features)
# 
# -  Attribute information
# 
#     -  ID number
#     
#     -  Diagnosis(M= Malignant, B=Benign)
#     
# __(https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)__
#  
# 

# ---

# ## Step 2: Importing Data

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


from sklearn.datasets import load_breast_cancer 


# In[10]:


cancer = load_breast_cancer()


# In[6]:


cancer


# In[8]:


# Here are the keys that are available in our dataset
cancer.keys()


# In[10]:


# Let's explore the keys in detail
print(cancer['DESCR'])
# It gives the minimum information about the data


# In[11]:


print(cancer['target_names'])


# In[13]:


print(cancer['target'])
# We can see the target is in 0,1.


# In[14]:


print(cancer['feature_names'])


# In[16]:


cancer['data'].shape

# So here we have 569 rows, 30 columns.


# In[11]:


# Let's prepare a dataframe to get a better view
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))


# In[12]:


df_cancer.head()


# In[19]:


df_cancer.tail()


# ---

# # Step 3: Visualizing the Data

# ### Let's view the relationship between the variables using Pairplot from seaborn.
# 

# In[23]:


sns.pairplot(df_cancer,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness'])


# Though from the above plot we can see the rellationship between the variables, but it is not so clear with target variables.So, to view how these are relating to the target variable, let's draw another plot with hue.

# In[24]:


sns.pairplot(df_cancer,hue='target',vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness'])


# From the above plot we can see the target as 0,1. which means malignant or benign.
# 
# The below plot shows how many malignant cases are there and how many benign.

# In[25]:


sns.countplot(df_cancer['target'])


# Now, let's see the scatter plot between one of the pairs from pair plot.

# In[32]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='mean radius', y='mean smoothness',hue='target', data=df_cancer )


# Now let's look into the **correlation** with seaborn heatmap function 

# In[14]:


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)


# ---

# # Step 4: Model Training(Finding a Problem Solution)

# To train our model first we need to define X and Y variables.

# In[15]:


X = df_cancer.drop(['target'], axis=1)


# In[16]:


X


# In[17]:


y = df_cancer['target']


# In[18]:


y


# Let's split our data into train and test using sklearn.

# In[19]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[33]:


X_train


# In[34]:


y_train


# In[35]:


X_test


# In[36]:


y_test


# In[37]:


from sklearn.svm import SVC


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix


# In[63]:


svc_model = SVC()


# In[64]:


# Fitting our model
svc_model.fit(X_train, y_train)


# ---

# # Step 5: Evaluating the Model

# In[79]:


y_predict_old = svc_model.predict(X_test)


# In[80]:


y_predict_old


# In[81]:


cm_old = confusion_matrix(y_test, y_predict_old)


# In[82]:


# View confusion matrix using seaborn
sns.heatmap(cm_old, annot=True)


# In[93]:


print(classification_report(y_test, y_predict_old))


# From the confusion matrix we can see that we have **48** misclassified values. 
# 
# To avoid this and **improve** our model we need to consider some parameters, namely:
# 
# -  Data Normalization
# -  SVM Parameters
#     - C Parameter
#     
#     - Gamma Paramtere

# # Step 6: Improving the Model  ~~~Part 1

# In[60]:


# Normalizing the data
# normalized = (x-min(x))/(max(x)-min(x))
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train-min_train)/range_train


# In[61]:


X_train_scaled


# Let's plot the scatter plot using any pair varaible in data with seaborn to see what has changed before and after normalization

# In[68]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[69]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# The values in y-axis are from 0-1. Thus we got perfect normalised data. Let's do it for test data also.

# In[70]:


# Normalizing the data
# normalized = (x-min(x))/(max(x)-min(x))
min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test-min_test)/range_test


# In[71]:


sns.scatterplot(x = X_test_scaled['mean area'], y = X_test_scaled['mean smoothness'], hue = y_test)


# Let's fit the model again with normalised data

# In[72]:


# Fitting our model
svc_model.fit(X_train_scaled, y_train)


# In[73]:


y_predict = svc_model.predict(X_test_scaled)


# **Let's draw confusion matrix with normalised data using seaborn**

# In[74]:


cm = confusion_matrix(y_test, y_predict)


# In[88]:


sns.heatmap(cm_old, annot=True)


# In[89]:


sns.heatmap(cm,annot=True)


# In[90]:


print(classification_report(y_test, y_predict))


# Considering Old and New Heatmap Confusion Matrices we can see that In Normalised data there are only **5** Errors and also only Type 1 error, which means we predicted that patient has cancer but in the true case he does not  have cancer.
# 
# Type 2 error, which means we predicted that patient has no cancer but in true case he does have cancer.
# 
# While considering both the classification reports we can see how huge the precision rate has changed.

# ---

# # Step 6: Improving the Model~~~ Part 2

# In[98]:


from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV


# In[95]:


param_grid = {'C':[0.1,1,10,100], "gamma":[1,0.1,0.1,0.01], "kernel":['rbf']}


# In[101]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=4)


# In[102]:


grid.fit(X_train_scaled, y_train)


# In[103]:


grid.best_params_


# In[105]:


grid_predictions = grid.predict(X_test_scaled)


# In[106]:


cm_grid = confusion_matrix(y_test, grid_predictions)


# In[107]:


sns.heatmap(cm_grid, annot=True)


# #### Here we got the best results with only 3 errors. 
# 
# **Let's see classification report and check the accuracy.**

# In[108]:


print(classification_report(y_test, grid_predictions))


# **Great!! We got 97% accuracy which is best fit for the model.**

# ---

# # Conclusion

# -  **Machine learning techniques(SVM) was able to classify tumors into Malignant and Benign with 97% accuracy.**
# 
# -  **The Technique can rapidly evaluate cancer detections and classify them in an automated fashion.**
# 
# -  **With this an early study can be conducted and can be cured many patients.**
# 
# -  **This technique can be further improved by combining Computer Vision with ML to directly classify cancer using tissue images, which can be done through Neural Networks.**
