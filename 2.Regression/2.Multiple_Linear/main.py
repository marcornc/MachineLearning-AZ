#       Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#       Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#       Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

#       Splitting Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#       Training Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Build multilple linear model
regressor.fit(X_train, y_train) # Train the model on the training set

#       Predicting Test Results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) # To have 2 decimal 
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)) # Concatenate verticlally the predict profit with the real one (test)

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)