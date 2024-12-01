# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Iris dataset
df = pd.read_csv('data4.csv')
# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1] 
# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Apply feature scaling on the training and test sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Print the scaled training and test sets
print(X_train)
print(X_test)