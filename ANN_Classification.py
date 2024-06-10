import pandas as pd
import numpy as np
cancer = pd.read_csv("R:\DP\ANN\ovariantotal.csv")
cancer.head()
# Or cancer.shape


#Spilt the features and label
X=cancer.drop('TYPE', axis=1)
y=cancer['TYPE']
print(X)

#Feature scaling for values n csv - 1). MinMax scalar and 2). Normalization
from sklearn import preprocessing
min_max_scaler=preprocessing.MinMaxScaler()
X_scale=min_max_scaler.fit_transform(X)
print(X_scale)


#Split the data for training and testing - train_test_split() & test_size=0.20(20% for testing), for splitting the dataset for testing and training, its to be done by randomizing the data - random_state=0/1(true/false)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_scale, y, test_size=0.20, random_state=0)


#DeepLearning=> keras, tensorflow and pytorch packages
#DEEPLEARNING satrts from here
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense


#dense-> All interconnected

model=Sequential()

# hidden layers

model.add(Dense(32, activation='relu', input_shape=(49,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(34, activation='relu'))
model.add(Dense(34, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))

#complexity of the model can be increased b bby having more hidden layers


#output layers
model.add(Dense(1, activation='sigmoid'))

#compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


#SGD with momentum

opt= keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10)


y_pred=model.predict(X_test)
y_pred#print
#convert the predicted real numbers in 0 and 1
y_pred=np.where(y_pred>0.5, 1, 0)
y_pred
import numpy as np
np.column_stack((y_pred, y_test))


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))