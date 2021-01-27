import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['PRICE'] = boston.target

from sklearn.model_selection import train_test_split

X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = bos['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 65)

from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X_train,y_train)

coeff_diff= pd.DataFrame(lin.coef_, X.columns, columns= ['Coefficient'])

predictions = lin.predict(X_test)
sns.displot((y_test-predictions),bins=40, kde=True)
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('MAE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))