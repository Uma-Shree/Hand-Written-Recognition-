#!/usr/bin/env python
# coding: utf-8

# Recognizing Hand-Written digits

# In[7]:


#importing necessary libraries
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import svm ,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
digits=load_digits()


# In[9]:


print("Image data shape",digits.data.shape)
print("Label data shape",digits.target.shape)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' %label , fontsize =20)
    


# In[ ]:





# Dividing dataset into Training and Test set

# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data, digits.target, test_size=0.25,random_state=2)


# In[35]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# importing the Logistic Regression model , instantiating it and  training it

# In[36]:


from sklearn.linear_model import LogisticRegression
logisticRegr=LogisticRegression()
logisticRegr.fit(x_train,y_train)


# In[37]:


print(logisticRegr.predict(x_test[0].reshape(1,-1)))


# In[40]:


print(x_test[0].reshape(1,-1))


# In[41]:


logisticRegr.predict(x_test[0:10])


# Predicting for the entire dataset

# In[42]:


predictions=logisticRegr.predict(x_test)


# Determing the ACCURACY of the model

# In[44]:


score=logisticRegr.score(x_test, y_test)
print(score)


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[57]:


cm=metrics.confusion_matrix(y_test,predictions)
print(cm)


# In[ ]:





# In[ ]:





# In[ ]:





# Representing the confusion matrix in a heat map

# In[59]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True , fmt=".3f", linewidth=.5, square=True , cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title='Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)


# In[68]:


index=0
classifiedIndex=[]
for predict,actual in zip(predictions,y_test):
    if predict==actual:
        classifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4,plotIndex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}, " .format(predictions[wrong], y_test[wrong]), fontsize=20)


# In[66]:


index=0
misclassifiedIndex=[]
for predict,actual in zip(predictions,y_test):
    if predict!=actual:
        misclassifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1,4,plotIndex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}, " .format(predictions[wrong], y_test[wrong]), fontsize=20)


# In[ ]:




