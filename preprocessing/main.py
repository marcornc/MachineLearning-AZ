import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
