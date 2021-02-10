import pandas as pd
from perceptron import Perceprton
from decision_regions import plot_decision_regions

data = pd.read_csv('iris.csv')
data = data[data['Species'].isin(['Iris-setosa','Iris-versicolor'])]
data = data.drop('Id',axis=1)
data

import matplotlib.pyplot as plt
import numpy as np

y = data.loc[:,'Species'].values # .values -> array instead of pd.Series()
y = np.where(y == 'Iris-setosa',-1,1) # if y == 'Iris-setosa' y=-1 else y=1

X = data.iloc[:,[0,2]].values

plt.scatter(X[:50, 0],X[:50, 1], color='red', marker = 'o', label='setosa')
plt.scatter(X[50:, 0],X[50:, 1], color='blue', marker = 'o', label='versicolor')
plt.legend()

ppn = Perceprton(eta = 0.1,n_iter=5)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker='o')

plot_decision_regions(X,y,classifier=ppn)