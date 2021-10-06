#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


data = pd.read_csv("diabetes.csv")


# In[26]:



data.shape


# In[27]:


data.head(5)


# In[28]:


# check if any null value is present
data.isnull().values.any()


# In[29]:


## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[30]:


data.corr()


# In[31]:


diabetes_map = {True: 1, False: 0}


# In[32]:



data['Outcome'] = data['Outcome'].map(diabetes_map)


# In[33]:


data.head(5)


# In[34]:


diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])


# In[35]:


(diabetes_true_count,diabetes_false_count)


# In[36]:


## Train Test Split

from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'SkinThickness']
predicted_class = ['Outcome']


# In[37]:


X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[22]:


print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['BMI'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['Age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['SkinThickness'] == 0])))


# In[42]:


#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[43]:



## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())


# In[45]:


##prediction
predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[ ]:


#################### SVM ##############################################


# In[51]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[52]:


data.groupby('Outcome').mean()


# In[53]:


# separating the data and labels
X = data.drop(columns = 'Outcome', axis=1)
Y = data['Outcome']


# In[54]:


print(X)


# In[55]:


print(Y)


# In[56]:


#Data Standardization
scaler = StandardScaler()


# In[57]:


scaler.fit(X)


# In[58]:


standardized_data = scaler.transform(X)
print(standardized_data)


# In[59]:


X = standardized_data
Y = data['Outcome']


# In[60]:


print(X)
print(Y)


# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[62]:


print(X.shape, X_train.shape, X_test.shape)


# In[63]:


classifier = svm.SVC(kernel='linear')


# In[64]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[65]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[66]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[67]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[68]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[69]:


#Making a Predictive System

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




