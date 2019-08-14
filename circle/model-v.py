#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(sys.argv[1])

X = df[['circle_y']]
Y = df['v']
polynomial_features = PolynomialFeatures(degree=4)
X_poly = polynomial_features.fit_transform(X)
lr = LinearRegression()
lr.fit(X_poly, Y)
print('coef=', lr.coef_)
print('intercept=', lr.intercept_)

plt.scatter(X, Y)
plt.plot(X, lr.predict(polynomial_features.fit_transform(X)), color='red')
plt.xlabel('circle_y')
plt.ylabel('v')
plt.show()
