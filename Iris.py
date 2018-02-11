
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pd.read_csv('iris.data',names=names)

x=dataset.iloc[:,0:4]
y=dataset.iloc[:,4]
#op=pd.get_dummies(data=dataset['class'],drop_first=True)

#print dimension
print(dataset.shape)


#print head 
print (dataset.head())

#describe
print(dataset.describe())

#info
print(dataset.info())

#to see classwise dist
print(dataset.groupby('class'))




get_ipython().magic('matplotlib inline')


# In[2]:


dataset.plot(kind='box',subplots=True,layout=(2,2))
plt.show()


# In[3]:


dataset.plot(kind='hist',subplots=True,layout=(2,2))
plt.show()


# In[14]:


sns.pairplot(data=dataset)


# In[4]:



from sklearn.cross_validation import train_test_split
validation_size=0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=validation_size, random_state=7)



# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[38]:


model=[]
model.append(('LR', LogisticRegression()))
model.append(('LDA',LinearDiscriminantAnalysis()))
model.append(('KNN',KNeighborsClassifier()))
model.append(('naive',GaussianNB()))
model.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
model.append(('SVM',SVC()))
result=[]
names=[]
for name,mode in model:
    kfold=model_selection.KFold(n_splits=10,random_state=7)
    var=model_selection.cross_val_score(mode,X_train,Y_train,cv=kfold,scoring='accuracy')
    result.append(var)
    names.append(name)
    msg = "%s: %f (%f)" % (name, var.mean(), var.std())
    print(msg)

    
 

              


# In[ ]:





# In[91]:


svmclassifier=SVC(kernel='linear')
svmclassifier.fit(X_train,Y_train)
pred=svmclassifier.predict(X_validation)
print(accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation, pred))
print(classification_report(Y_validation, pred))

