#######
# I don't need the Feature Scaling becose the Coefficient of the equescion will take care of it
#######
# Ordinary Least Squares
## to find the best Linear Regression

# Import Libraries
import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('Salary_data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting Data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set result
y_pred = regressor.predict(X_test)

# Visualise Training set 
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salari vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# Visualise Test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salari vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making Single Prediction
print(regressor.predict([[12]]))

# Getting Linear Regression Equation with Coefficients Values
print(regressor.coef_)
print(regressor.intercept_)
