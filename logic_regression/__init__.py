import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam 1', 'exam2', 'admitted'])
# print(data.head())

# 用isin索引
positive = data[data['admitted'].isin([0])]

