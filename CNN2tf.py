import tensorflow as tf 
import os
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.random.seed(10)
from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()

X_train=X_train.reshape(-1,32,32,3)/255 #normalize
X_test=X_test.reshape(-1,32,32,3)/255 #normliaze
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10) #將label 0-9 轉換成 1x10 矩陣

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def biase_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2D(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
xs=tf.placeholder(tf.float32,[None,1024])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
#conv1
x_image=tf.reshape(xs,[-1,32,32,3])
W_conv1=weight_variable([5,5,3,32])
b_conv1=biase_variable([32])
h_conv1=tf.nn.relu(conv2D(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#conv2
W_conv2=weight_variable([5,5,32,64])
b_conv2=biase_variable([64])
h_conv2=tf.nn.relu(conv2D(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#func1
W_fc1=weight_variable([8*8*64,1024])
b_fc1=biase_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#func2

W_fc2=weight_variable([1024,10])
b_fc2=biase_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:0.75})
	correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:0.75})
	return result

epoch=10
batch_size=128
train_datasize,_,_,_=X_train.shape
period=round(train_datasize/batch_size)
for i in range(epoch):
	idxs=np.random.permutation(train_datasize)
	x_random=X_train[idxs]
	y_random=y_train[idxs]
	for j in range(period):
		batch_xs=x_random[j*batch_size:(j+1)*batch_size]
		batch_ys=y_random[j*batch_size:(j+1)*batch_size]
		batch_xs=batch_xs.reshape([-1,1024])
		sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.75})
		print(compute_accuracy(batch_xs,batch_ys))
#for i in range(100):
#	batch_xs,batch_ys=next_batch(X_train,y_train,128)
#	batch_xs=np.array(batch_xs)
#	batch_xs=batch_xs.reshape([-1,1024])
#	sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.75})
#	if i%50==0:
#		print(compute_accuracy(batch_xs,batch_ys))
