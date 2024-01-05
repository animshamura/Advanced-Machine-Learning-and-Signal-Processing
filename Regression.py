# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
  
#loading iris dataset from seaborn
df = sns.load_dataset("iris")
 
#separate feature and target
data= df.values
x= data[:,0:4] # independent variable
y = data[:,4] # dependent variable
 
#importing train_test_split
from sklearn.model_selection import train_test_split   
 #dividing into training and testing set
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 0.25)
 
#importing  Logistic Regression and fitting the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
 
#predicting the accuracy score
pred = lr.predict(x_test) # to x predict test set
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
