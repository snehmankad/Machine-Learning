import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# dataframe of train and test
train = pd.read_csv("C:\\Users\\USER\\Desktop\\acm_task_dataset-master\\train.csv")
test = pd.read_csv("C:\\Users\\USER\\Desktop\\acm_task_dataset-master\\test.csv")

# because of error shown NaN 
train.dropna(inplace=True)
test.dropna(inplace=True)

# seperating feature and label
X_train = train['x']
X_train = np.array(X_train)
y_train = train['y']
y_train = np.array(y_train)

X_train = preprocessing.scale(X_train)

# to convert 1D array to 2D array
X_train = X_train.reshape(-1,1)

X_test = test['x']
X_test = np.array(X_test)
y_test = test['y']
y_test = np.array(y_test)

X_test = preprocessing.scale(X_test)

X_test = X_test.reshape(-1,1)


clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

print(confidence)
print(clf.predict(X_test[:5]))

