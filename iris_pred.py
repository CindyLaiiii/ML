#!/usr/bin/env python
# coding: utf-8

# In[16]:


# load the iris data
from sklearn.datasets import load_iris
iris = load_iris()


# In[17]:


# store the feature matrix(X):input, and response vector(y):output (pre labeled answers)
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature names",feature_names)
print("Target names",target_names)


# In[18]:


# split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


# In[19]:


#KNN Classifier Try changing the n_neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# knn2 = DecisionTreeClassifier()
# knn2.fit(X_train,y_train)

#make prediction
y_pred = knn.predict(X_test)


# In[20]:


#Accuracy of our model based on our test output and prediction output
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[22]:


#Model persistance is important. Net time we use this saved model to make prediction.
import joblib
joblib.dump(knn,'mlbrain.joblib')


# In[24]:


#Load our model
model = joblib.load('mlbrain.joblib')

model.predict(X_test)
sample = [[3,5,4,2],[2,3,5,4]]
predictions = model.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print("predictions:",pred_species)


# In[26]:


# Data Visualization1
from sklearn.datasets import load_iris
iris = load_iris()
import matplotlib.pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# colorbar with the Iris target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

# chart configurations
plt.figure(figsize=(5,4))
plt.scatter(iris.data[:,x_index],iris.data[:,y_index],c=iris.target)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


# In[31]:


# Data Visualization2
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

# chart configurations
plt.scatter(features[2],features[3],alpha=0.2,s=100*features[3],c=iris.target,cmap='viridis')

plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.colorbar(ticks=[0,1,2],format=formatter)


# In[ ]:




