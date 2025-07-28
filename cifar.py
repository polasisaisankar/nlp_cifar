from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#import data 
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Build Architecture
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(1026,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train
history = model.fit(x_train,y_train,epochs = 3,batch_size = 64,validation_split = 0.2)


#evaluate
test_accuracy,loss = model.evaluate(x_test,y_test)
print(f'test_accuracy:{test_accuracy}')
print(f'loss:{loss}')

#visualization
print(type(history))
print(history.history.keys())
print(history.history)
plt.plot(history.history['accuracy'],color = 'blue', label = 'train_accuracy')
plt.plot(history.history['val_accuracy'],color = 'red',label='val_accuracy')
plt.legend()
plt.title('Epochs vs  Accuracy')
#plt.show()

#exp:1 optimizer : adam,batch_size:64
#model.add(Dense(128,activation='relu'))
#model.add(Dense(10,activation='softmax'))
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,epochs = 10,batch_size = 64)

#exp:2 optimizer : sgd,batch_size:64
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs = 10,batch_size = 64)

#exp:3 optimizer : sgd,batch_size:64,one more layer(256 nuerons)
#model.add(Dense(256,activation='relu'))
#model.add(Dense(128,activation='softmax'))
#model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,epochs = 10,batch_size = 64)

#exp:4 optimizer : adam,batch_size:64,one more layer(256 nuerons)
#model.add(Dense(256,activation='relu'))
#model.add(Dense(128,activation='relu'))
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,epochs = 10,batch_size = 64)


#exp:5 optimizer : adam,batch_size:64,epochs increase from 10 to 30
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,epochs = 30,batch_size = 64)