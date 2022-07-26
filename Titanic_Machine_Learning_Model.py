#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

for dirname, _, filenames in os.walk("D:\Practice Data Sets\Titanic Dataset-Machine Learning Model"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_data = pd.read_csv("D:/Practice Data Sets/Titanic Dataset-Machine Learning Model/train.csv",header=0,encoding='unicode_escape')
test_data = pd.read_csv("D:/Practice Data Sets/Titanic Dataset-Machine Learning Model/test.csv",header=0,encoding='unicode_escape')


# In[3]:


train_data.head()


# In[4]:


train_data.describe()


# In[5]:


#Check Null Values:
#---------------------
train_data.isnull().sum()


# In[6]:


#Fill the null values:
#------------------------
#train
#------
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Cabin'] = train_data['Cabin'].fillna('NotSet')
train_data['Embarked'] = train_data['Embarked'].fillna('N')

#test
#-----
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Cabin'] = test_data['Cabin'].fillna('NotSet')
test_data['Embarked'] = test_data['Embarked'].fillna('N')


# In[7]:


#Age distribution:
#------------------
sns.set(color_codes=True)
sns.distplot(train_data['Age'])


# In[8]:


#Survival by sex:
#-----------------
train_data[['Sex','Survived']].groupby('Sex').sum()


# In[9]:


df_survivedsex=train_data[['Sex','Survived']]

def superviviente (row):
    if row['Survived'] == 1:
        return 0
    else:
        return 1

df_survivedsex['NotSurvived'] = df_survivedsex.apply (lambda row: superviviente(row), axis=1)

df_survivedsex = df_survivedsex.groupby('Sex').sum()

df_survivedsex.plot(kind = 'bar',stacked = 'True',alpha = 0.4,width = 0.9,figsize=(7,6))


# In[10]:


#Class:
#--------
#Survival by class:
#-------------------
train_data[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[11]:


df_survivedclass=train_data[['Pclass','Survived']]
df_survivedclass['NotSurvived']=df_survivedclass.apply (lambda row: superviviente(row), axis=1)
df_survivedclass=df_survivedclass.groupby('Pclass').sum()
df_survivedclass.plot(kind='bar',stacked='True',alpha=0.4,width=0.9,figsize=(7,9))


# In[12]:


train_data


# In[13]:


#Heatmap:
#----------
df_map = train_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
df_map=pd.get_dummies(df_map,columns=['Sex'])
ax=sns.heatmap(df_map.corr(),cmap="PiYG")


# In[14]:


#Models
#--------
#Delete Id, Ticket and name. We think they are not relevant
#------------------------------------------------------------
train_data_2 = train_data.drop(["PassengerId","Ticket","Name"], axis=1)
test_data_2 = test_data.drop(["PassengerId","Ticket","Name"], axis=1)

#Categories to columns
#-----------------------
train_data_2 = pd.get_dummies(train_data_2, columns=["Sex", "Cabin", "Embarked"])
test_data_2 = pd.get_dummies(test_data_2, columns=["Sex", "Cabin", "Embarked"])

#FillNa  wiith mean
#--------------------
train_data_2 = train_data_2.fillna(train_data_2.mean())
test_data_2 = test_data_2.fillna(test_data_2.mean())

test_data_2 = test_data_2.reindex(columns = train_data_2.columns, fill_value=0)
test_data_2=test_data_2.drop(['Survived'], axis=1)

#separate the dataset into training and validation
#--------------------------------------------------
from sklearn.model_selection import train_test_split
#X, Y = imputeddata.drop(["Survived"], axis=1), imputeddata["Survived"]
#------------------------------------------------------------------------
Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_data_2.drop(["Survived"], axis=1), train_data_2["Survived"], test_size=0.33)


# In[15]:


from sklearn.tree import DecisionTreeClassifier

#create an instance of decisiontree
#-----------------------------------
decisiontree = DecisionTreeClassifier()


# In[16]:


from sklearn import tree
from graphviz import Source

#Function that returns the visualization:
#------------------------------------------
def plottree(decisiontree, features_names=None, class_names=None):
    dot_data = tree.export_graphviz(
        decisiontree,
        out_file=None,
        filled=True,
        rounded=True,
        rotate=True,
        feature_names=features_names,
        class_names=class_names
    )
    return Source(dot_data)


# In[17]:


#Train the model:
#-----------------
decisiontree = DecisionTreeClassifier()
decisiontree.fit(Xtrain, Ytrain)


# In[18]:


#Accuracy:
#---------
from sklearn.metrics import accuracy_score

Ypred = decisiontree.predict(Xtest)
decision_tree_model_acc = accuracy_score(Ypred,Ytest) * 100

print("Accuracy:",decision_tree_model_acc)


# In[19]:


#Show the generated tree
#-----------------------
plottree(decisiontree, features_names=Xtrain.columns, class_names=["Deceased","Survived"])


# In[21]:


#Random Forest:
#--------------
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(criterion = 'gini',
                                            n_estimators=1750,
                                            max_depth=7,
                                            min_samples_leaf=6,
                                            max_features='auto',
                                            verbose=1,
                                            random_state=3)
random_forest_model.fit(Xtrain, Ytrain)
Ypred = random_forest_model.predict(Xtest)
random_forest_model_acc=accuracy_score(Ypred, Ytest) * 100
print("Accuracy:", random_forest_model_acc)


# In[22]:


fn=train_data_2.drop(["Survived"],axis=1).columns.to_numpy()
cn=np.array(['Decease','Survived'])
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(5,5),dpi=800)
tree.plot_tree(random_forest_model.estimators_[0],feature_names=fn,class_names=cn,filled=True)


# In[24]:


#K-Nearest-Neighbor:
#---------------------
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1,26)
scores=[]

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, Ytrain)
    Ypred = knn.predict(Xtest)
    scores.append(metrics.accuracy_score(Ytest,Ypred))
print(scores)


# In[25]:


import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')


# In[26]:


#K value equal 9 has the highest accuracy rate.
#--------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(Xtrain,Ytrain)
Ypred=knn.predict(Xtest)
print(metrics.accuracy_score(Ytest,Ypred))


# In[27]:


#TEST DATA
#----------
Ypred = knn.predict(test_data_2)
Submission = pd.DataFrame({"PassengerID":test_data["PassengerId"],"Survived":Ypred})
Submission.to_csv('my_submission.csv',index=False)


# In[ ]:




