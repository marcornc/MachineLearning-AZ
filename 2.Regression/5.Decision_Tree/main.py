#       Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#       Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#       Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(random_state=0)
regr.fit(X, y)

#       Predicting a new result
print(regr.predict([[6.5]]))

#       Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='red')
plt.plot(X_grid, regr.predict(X_grid), c='blue')
plt.show()