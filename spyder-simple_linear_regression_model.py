# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data = pd .read_csv(r'C:\Users\rohit\Downloads\Salary_Data.csv')
data

# Split the dataset
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.20, random_state=0)

# Reshape the dataset values
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# model building pipeline

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Visualization
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_inter =  regressor.intercept_
print(c_inter)

y_15 = m_slope*20 + c_inter
print(y_15)

print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

data.mean()

data['Salary'].mean()

data.median()

data['Salary'].mode()

data.describe()

data.var()

data.std()

from scipy.stats import variation
variation(data.values)

variation(data['Salary'])

data.corr()
data['Salary'].corr(data['YearsExperience'])

data.skew()

data['Salary'].skew()

data.sem()

# Z-score

import scipy.stats as stats

data.apply(stats.zscore) # 

stats.zscore(data['Salary']) # this will give us Z-score of that particular column

a = data.shape[0]
b = data.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom)

# Sum of Square Regression (SSR)

y_mean = np.mean(y)

SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# Sum of Squares Error (SSE)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# Sum of Squares Total (SST)

mean_total = np.mean(data.values) # here data.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((data.values-mean_total)**2)
print(SST)

# R2

r_square = 1 - SSR/SST
r_square

from sklearn.metrics import mean_squared_error

import pickle
filename = 'regressor.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

