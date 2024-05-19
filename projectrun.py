#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your data
# Assuming your data is in a DataFrame named 'df'
# You might need to replace 'path_to_your_data.csv' with the actual path to your data file
df = pd.read_csv('TTest.csv')


df['Gender'].replace({'Male' : 1 ,'Female' : 0},inplace=True)
df['Customer Type'].replace({'Loyal Customer' : 1 ,'disloyal Customer':0},inplace=True) 
df['Type of Travel'].replace({'Business travel':1,'Personal Travel':0},inplace=True)
df['Class'].replace({'Business':1,'Eco Plus':2,'Eco':3},inplace=True)

df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(),inplace = True)

data = df.drop(['Unnamed: 0'],axis=1)

x = data.iloc[:,:-1].values
y = data.iloc[:,15].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y)


# Train and save Perceptron model
log_model_re = LogisticRegression(max_iter=1000)
log_model_re.fit(x_train, y_train)

pickle.dump(log_model_re, open('log_model_re.pkl', 'wb'))