import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN,Activation,Dense
from keras.optimizers import Adam

time_steps=28 #每一次讀取一列 共有28列
input_size=28 #每一次讀取一列28個pixel
batch_size=50
batch_index=0 #生成數據
output_size=10 #MNIST classes(0-9 digits)
cell_size=50 #RNN的hidden layer
lr=0.001
#X shape (60,000,28x28),y shape (10,000, )
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(-1,28,28)/255 #normalize 變成0~1的區間
X_test=X_test.reshape(-1,28,28)/255 #normalize 變成0~1的區間
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

model=Sequential()

#RNN cell
model.add(SimpleRNN(batch_input_shape=(None,time_steps,input_size),output_dim=cell_size,unroll=True))
#output layer
model.add(Dense(output_size))
model.add(Activation('softmax'))
#optimizer
adam=Adam(lr)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#training
for step in range(4001):
	#data shape=(batch_num,steps,input/outputs)
	X_batch=X_train[batch_index:batch_size+batch_index,:,:]#第一個':' num of time steps 第二個':' num of input size
	Y_batch=y_train[batch_index:batch_size+batch_index,:]
	cost=model.train_on_batch(X_batch,Y_batch)
	batch_index += batch_size
	batch_index=0 if batch_index>=X_train.shape[0] else batch_index #如果batch_index累加超過總體Sample個數 則從0重新開始
	if step %500 ==0:
		cost,accuracy=model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
		print('test cost',cost,'test accuracy',accuracy)
