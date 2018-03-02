from sklearn.datasets import load_breast_cancer
from sklearn import metrics

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

# importing the train and test data
cancer = load_breast_cancer()

x_train = cancer.data[:340]
y_train = cancer.target[:340]

x_test = cancer.data[340:]
y_test = cancer.target[340:]

# defining the model
model = Sequential()

# model.add(Dense(15, input_dim=30, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) # output layer
# -------
# model.add(Dense(15, input_dim=30, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Once we have defined the model we will then compile the model by supplying the necessary optimizer, loss function, and the metric on which we want to evaluate the model performance.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=50)

predictions = model.predict_classes(x_test)

print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))
print('Everything is running')
