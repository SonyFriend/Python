import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPool2D,Flatten
from keras.optimizers import Adam

(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(-1,1,28,28)
X_test=X_test.reshape(-1,1,28,28)
y_train=np_utils.to_categorical(y_train, num_classes=10)
y_test=np_utils.to_categorical(y_test, num_classes=10)

model=Sequential()

model.add(Convolution2D(nb_filter=32,
	nb_row=5,
	nb_col=5,
	border_mode='same',
	input_shape=(1,28,28)
	))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),border_mode='same'))

model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),border_mode='same'))
model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam=Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch=1,batch_size=32)

print('\nTesting ------------')
loss,accuracy=model.evaluate(X_test,y_test)

print('\ntest loss', loss)
print('\ntest accuracy', accuracy)