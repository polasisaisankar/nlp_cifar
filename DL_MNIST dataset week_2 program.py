import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

#Load the data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#print(X_train.shape)

#normalize the data

#categorical 
#print(y_train[0])
#y_train=y_train.astype('float32')/255.0
#y_test=y_test.astype('float32')/255.0

#one-hot encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#build the architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compile model
model.compile(optimizer='sgd',loss='categorical_crossentropy')

#train the model
model.fit(X_train,y_train,epochs=10,batch_size=32)

#Evaluate
model.evaluate(X_test,y_test)


model.compile(optimizer='adam',loss='categorical_crossentropy')
model.fit(X_train,y_train,epochs=50,batch_size=64)
model.evaluate(X_test,y_test)