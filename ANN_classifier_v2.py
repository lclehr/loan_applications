#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#above all relevant libraries are imported
# importing the libraries


# In[2]:


#importing data set
loan_data = pd.read_csv('/Users/linus/Documents/BA_Portfolio/Auto_Encode/train.csv')
loan_data


# In[3]:


#NA values have to be dropped to make the model work
loan_data = loan_data.dropna()

#encoding is performed for all binary non numerical columns as most ML models required numerical inputs
# Encoding dictionaries for binary variables are specified
gender_dict = {'Male': 0, 'Female': 1}
married_dict = {'No': 0, 'Yes': 1}
education_dict = {'Not Graduate': 0, 'Graduate': 1}
self_employed_dict = {'No': 0, 'Yes': 1}
Loan_Status_dict = {'Y' : 1, 'N' : 0}

#the dictionaries have been mapped to the columns
loan_data['Gender'] = loan_data['Gender'].map(gender_dict)
loan_data['Married'] = loan_data['Married'].map(married_dict)
loan_data['Education'] = loan_data['Education'].map(education_dict)
loan_data['Self_Employed'] = loan_data['Self_Employed'].map(self_employed_dict)
loan_data['Loan_Status'] = loan_data['Loan_Status'].map(Loan_Status_dict)

#one hot encoding is used for the only non-binary column, because label encoding would only be used for a ranked variable
loan_data = pd.get_dummies(loan_data, columns=['Property_Area'])


#because dependents has numbers as string, this needs to be transformed
#for applicants with three or more dependents 3+ is in the origianl data set
#to be able to better process it, this is turned to 3
def encode_dependents(value):
    if value == '3+':
        return 3
    else:
        return int(value)

#now the encoder function has been applied to the column denoting the dependents
loan_data['Dependents'] = loan_data['Dependents'].apply(encode_dependents)
loan_data.to_csv('cleaned_loan_data.csv')
loan_data


# In[4]:


X = loan_data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Credit_History', 'Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']]
y = loan_data ['Loan_Status']
X


# In[5]:


# the features in the independent columns are scaled as part of preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# a standard test training split is performed with  the test size being 20% of the overall data set
# a random state is specified for reproduceability
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural network architecture
# the input dimension is modeeled after the shape of the training data set, getting the number of columns by using .shape()
input_dim = X_train.shape[1]

#below the three neuron layers are specified
#with a number of three layers this network can be classified as a deep learning ANN

#the shape of the input layer has been set to match the input dimension specified earlier
input_layer = Input(shape=(input_dim,))

# specification of the hidden layer with 64 neurons (this was chosen arbitrarily)
hidden_layer = Dense(64, activation='relu')(input_layer) 

# Since binary classification is done, the output layer has a single neuron.
#the output of this neuron will be a value between 0 and 1, representing the probability of the loan being approved
output_layer = Dense(1, activation='sigmoid')(hidden_layer)  

#the model is specified with the input set to the input layer and the output to the output layer
model = Model(inputs=input_layer, outputs=output_layer) 

# Below the model has been compiled  
# binary crossentropy is chosen as a loss function for a binary classification problem
#as optimizer adam is chosen because is is computationally efficient and has little memory requirements
#the metric to be optimized in the model is accuracy, to achieve accurate predictions later on
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training of the model
#50 training runs (called epochs) are specified and batch size of 32
#for both parameters different combinations were tried but non lead to better accuracy than this combo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

#Evaluate the model
eval_loss, eval_accuracy = model.evaluate(X_test, y_test)
print(f"Evaluation loss: {eval_loss}")
print(f"Evaluation accuracy: {eval_accuracy}")

#


# In[6]:


loans_for_eval = pd.read_csv('/Users/linus/Documents/BA_Portfolio/Auto_Encode/test.csv')
loans_for_eval


# In[7]:


#same preprocessing as prior
#NA values have to be dropped to make the model work
loans_for_eval = loans_for_eval.dropna()

#the dictionaries have been mapped to the columns
loans_for_eval['Gender'] = loans_for_eval['Gender'].map(gender_dict)
loans_for_eval['Married'] = loans_for_eval['Married'].map(married_dict)
loans_for_eval['Education'] = loans_for_eval['Education'].map(education_dict)
loans_for_eval['Self_Employed'] = loans_for_eval['Self_Employed'].map(self_employed_dict)

loans_for_eval = pd.get_dummies(loans_for_eval, columns=['Property_Area'])

#because dependents has numbers as string, this needs to be transformed
#for applicants with three or more dependents 3+ is in the origianl data set
#to be able to better process it, this is turned to 3
def encode_dependents(value):
    if value == '3+':
        return 3
    else:
        return int(value)

#now the encoder function has been applied to the column denoting the dependents
loans_for_eval['Dependents'] = loans_for_eval['Dependents'].apply(encode_dependents)
loans_for_eval


# In[8]:


X_eval = loans_for_eval[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Credit_History', 'Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']]
X_eval_scaled = scaler.transform(X_eval)

# After the usual preprocessing the model predictions are calculated
predictions = model.predict(X_eval_scaled)

# Since this is a binary classification model, the model prediction returns the probability of the sample being in class 1
# To convert these probabilities to class labels, 0.5 is used as a threshhold. Theoretically any number between 0 and 1 could be used.
#if the bank would be extremly risk averse, it would make sense to set this treshold higher, to ensure less credit defaults 
predicted_classes = (predictions >= 0.5).astype(int)


#the predicted classes are now saved to the loans for evaluation data frame to get the status for each loan classification
loans_for_eval['Accepted'] = predicted_classes
loans_for_eval


# In[ ]:




