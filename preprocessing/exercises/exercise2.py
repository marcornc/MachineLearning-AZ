# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('data2.csv')

# Identify missing data (assumes that missing data is represented as NaN)
missing_values = dataset.isnull().sum()

# Print the number of missing entries in each column
print(missing_values)

# Configure an instance of the SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset = imputer.transform(dataset)

#Print your updated matrix of features
print(dataset)