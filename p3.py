# CORRELATION AND REGRESSION ANALYSIS
# a.scatter diagram calculating correlation coefficient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

df = pd.read_csv(r'C:\Users\user\Downloads\housing (1).csv')

print(df.head())

# scater diagram
import matplotlib.pyplot as plt

plt.scatter(df['sqft_living'], df['price'])
plt.title("Scatter diagram: sqft_living vs. price")
plt.xlabel('sqft_living')
plt.ylabel('price')

plt.show()

# correlation
correlation_coefficient=df['sqft_living'].corr(df['price'])
# print(f'Correlation_coefficient:(sqrt_living vs. price):{correlation_coefficient}')

# /linear regression
import statsmodels.api as sm
x_simple = sm.add_constant(df['sqft_living'])  # Add constant to the predictor
y_simple = df['price']  # Target variable
model_simple = sm.OLS(y_simple, x_simple).fit()  # Fit the OLS model
print(model_simple.summary())  # Print the model summary

# multi linear regression
x_multi=sm.add_constant(df[['sqft_living', 'bedrooms','bathrooms']])
y_multi=df['price']
model_multi=sm.OLS(y_multi, x_multi).fit()
print(model_multi.summary())

# prediction
prediction_simple=model_simple.predict(x_simple)
prediction_simple

x_multi=sm.add_constant(df[['sqft_living', 'bedrooms','bathrooms']])
y_multi=df['price']
model_multi=sm.OLS(y_multi, x_multi).fit()
prediction_multi=model_multi.predict(x_multi)
prediction_multipredictions_multi=model_multi.predict(x_multi)
prediction_multi

# c.fitting linear regression
from sklearn.model_selection import train_test_split
x_logistic = df[['sqft_living', 'waterfront']]  # Features
y_logistic = (df['price'] > df['price'].median()).astype(int)  # Target: 1 if price > median, else 0
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x_logistic, y_logistic, test_size=0.2, random_state=42)
print(x_logistic)
print(y_logistic)
# 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

logreg=LogisticRegression()
logreg.fit(x_train_log, y_train_log)
y_pred_log=logreg.predict(x_test_log)
y_pred_log

# /model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_log=accuracy_score(y_test_log, y_pred_log)

print(f'Accuracy (Logistic Regression): {accuracy_log}')
print(f'Confusion Matrix (Logistic Regression):\n{confusion_matrix}')

accuracy_log
