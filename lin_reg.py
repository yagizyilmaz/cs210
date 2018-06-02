# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# Importing the dataset
dataset = pd.read_csv('4.csv')

# Slicing
dataset = dataset[['ID', 'Community Area', 'Primary Type', 'Latitude', 'Longitude']].loc[dataset['Primary Type']
 .isin(['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT', 'OTHER OFFENSE', 'BURGLARY'
        , 'DECEPTIVE PRACTICE', 'MOTOR VEHICLE THEFT', 'ROBBERY'])]
dataset = dataset.reset_index(drop=True)

dataset = dataset.dropna()
# dataset.isnull().any().any()

dataset2 = pd.read_csv("new_set.csv")
finalset = pd.merge(dataset, dataset2, on='Community Area')
finalset = finalset[['Primary Type', 'High', 'Bachelor']]

X = finalset.iloc[:, 1:].values
y = finalset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Avoiding the Dummy Variable Trap
#y = y.iloc[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#y = np.append(arr = np.ones((1306201, 1)).astype(int), values = y, axis = 1)
X_opt = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
