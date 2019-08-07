from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
import cv2 
import os
from PIL import Image
import matplotlib.pyplot as plt
np.random.seed(10)
####################################################################
img=[]
img1=[]
filename=os.listdir("C:/Users/user/Downloads/pythonCode/animal")
for file in filename:
	img.append(np.array(Image.open("C:/Users/user/Downloads/pythonCode/animal/"+file)))
img=np.array(img)

for i in range(img.shape[0]):
	img[i]=cv2.resize(img[i],(32,32))
	img1.append(img[i])
img1=np.array(img1)

img1_y=np.array([[3],[3],[3],[3],[3],[5],[5],[5]])
img1=img1.reshape(-1,32,32,3)/255 #normliaze
img1_y=np_utils.to_categorical(img1_y,num_classes=10)
########################################################################
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
#def plot_image(image): #畫圖用
#	fig=plt.gcf()
#	fig.set_size_inches(4,4)
#	plt.imshow(image,cmap='binary')
#	plt.show()
#print(y_train[0])
#print(plot_image(X_train[0]))
X_train=X_train.reshape(-1,32,32,3)/255 #normalize
X_test=X_test.reshape(-1,32,32,3)/255 #normliaze
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10) #將label 0-9 轉換成 1x10 矩陣

model=Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))#每次訓練迭代時,會隨機在神經網路中放棄25%的神經元,避免overfitting
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Convolution2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))

model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#epoch(訓練週期) batch_size 每一批次資料量 每一批次128筆資料下去訓練 verbose=1 顯示訓練過程
train_history=model.fit(X_train,y_train,validation_split=0.2,epochs=15,batch_size=128,verbose=1)
accuracy=model.evaluate(X_test,y_test,verbose=1)
accuracy_animal=model.evaluate(img1,img1_y,verbose=1)
print(accuracy[1])
print(accuracy_animal[1])
pre_ani=model.predict_classes(img1)
print(pre_ani)
#prediction=model.predict_classes(X_test)
#print(prediction[0:10])

def show_train_history(train_history,train,validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel('train')
	plt.xlabel('Epoch')
	plt.legend(['train','validation'],loc='upper left')
	plt.show()

show_train_history(train_history,'acc','val_acc') #acc訓練的準確率 val_acc驗證的acc