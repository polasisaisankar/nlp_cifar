from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

#import data 
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Build Architecture
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(10,activation='softmax'))

#
model.compile(optimizer='adam',loss='categorical_crossentropy')

#train
model.fit(x_train,y_train,epochs = 10,batch_size = 64)

#evaluate
model.evaluate(x_test,y_test)
