#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(sys.argv[1])

lr = LinearRegression()
X = df[['circle_x']]
Y = df['sin_theta']
lr.fit(X, Y)
print('coef=', lr.coef_)
print('intercept=', lr.intercept_)

plt.scatter(X, Y)
plt.plot(X, lr.predict(X), color='red')
plt.xlabel('circle_x')
plt.ylabel('sin_theta')
plt.show()
