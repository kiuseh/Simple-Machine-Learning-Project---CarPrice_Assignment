# Importing Libraries
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv("(file path).csv")

# check for missing data
dataset.isna().sum()

# Dataset info
dataset.info()

# we convert categorical data into numbers
#Importing libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# encoder function
one_hot = OneHotEncoder()
#categorical features
categories = ["CarName", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem"]

# transform
transformer = ColumnTransformer([("one_hot", one_hot, categories)], remainder="passthrough")

# fitting
transformed_X = transformer.fit_transform(X)

# splitting train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

# fitting model 
# Importing regressor
from sklearn.ensemble import RandomForestRegressor

# fitting model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)

# prediction
y_preds = regressor.predict(X_test)

# evaluating model with r2_score
# Import r2_score
from sklearn.metrics import r2_score

# And result
r2_score(y_test, y_preds)
