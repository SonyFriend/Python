import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/')

def discriminator(images,reuse_variables=None):
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_variables) as scope:
		d_w1=tf.get_variable('d_w1',[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1=tf.get_variable('d_b1',[32],initializer=tf.constant_initializer(0))
		d1=tf.nn.conv2d(input=images,filter=d_w1,strides=[1,1,1,1],padding='SAME')
		d1=d1+d_b1
		d1=tf.nn.relu(d1)
		d1=tf.nn.avg_pool(d1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

		d_w2=tf.get_variable('d_w2',[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2=tf.get_variable('d_b2',[64],initializer=tf.constant_initializer(0))
		d2=tf.nn.conv2d(input=d1,filter=d_w2,strides=[1,1,1,1],padding='SAME')
		d2=d2+d_b2
		d2=tf.nn.relu(d2)
		d2=tf.nn.avg_pool(d2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

		d_w3=tf.get_variable('d_w3',[7*7*64,1024],initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3=tf.get_variable('d_b3',[1024],initializer=tf.constant_initializer(0))
		d3=tf.reshape(d2,[-1,7*7*64])
		d3=tf.matmul(d3,d_w3)+d_b3
		d3=tf.nn.relu(d3)

		d_w4=tf.get_variable('d_w4',[1024,1],initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4=tf.get_variable('d_b4',[1],initializer=tf.constant_initializer(0))
		d4=tf.matmul(d3,d_w4)+d_b4
		return d4

def generator(z,batch_size,z_dim):
	#from z_dim to 56*56 dimension
	g_w1=tf.get_variable('g_w1',[z_dim,3136],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b1=tf.get_variable('g_b1',[3136],initializer=tf.truncated_normal_initializer(stddev=0.02))
	g1=tf.matmul(z,g_w1)+g_b1
	g1=tf.reshape(g1,[-1,56,56,1])
	g1=tf.contrib.layers.batch_norm(g1,epsilon=1e-5,scope='bn1')
	g1=tf.nn.relu(g1)
	#Generate 50 features
	g_w2=tf.get_variable('g_w2',[3,3,1,z_dim/2],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b2=tf.get_variable('g_b2',[z_dim/2],initializer=tf.truncated_normal_initializer(stddev=0.02))
	g2=tf.nn.conv2d(g1,g_w2,strides=[1,2,2,1],padding='SAME')
	g2=g2+g_b2
	g2=tf.contrib.layers.batch_norm(g2,epsilon=1e-5,scope='bn2')
	g2=tf.nn.relu(g2)
	g2=tf.image.resize_images(g2,[56,56])

	g_w3=tf.get_variable('g_w3',[3,3,z_dim/2,z_dim/4],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b3=tf.get_variable('g_b3',[z_dim/4],initializer=tf.truncated_normal_initializer(stddev=0.02))
	g3=tf.nn.conv2d(g2,g_w3,strides=[1,2,2,1],padding='SAME')
	g3=g3+g_b3
	g3=tf.contrib.layers.batch_norm(g3,epsilon=1e-5,scope='b3')
	g3=tf.nn.relu(g3)
	g3=tf.image.resize_images(g3,[56,56])

	g_w4=tf.get_variable('g_w4',[1,1,z_dim/4,1],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b4=tf.get_variable('g_b4',[1],initializer=tf.truncated_normal_initializer(stddev=0.02))
	g4=tf.nn.conv2d(g3,g_w4,strides=[1,2,2,1],padding='SAME')
	g4=g4+g_b4
	g4=tf.sigmoid(g4)
	#Dim of g4:batch_size*28*28*1
	return g4

z_dimensions=100
batch_size=100
z_placeholder=tf.placeholder(tf.float32,[None,z_dimensions],name='z_placeholder')
x_placeholder=tf.placeholder(tf.float32,shape=[None,28,28,1],name='x_placeholder')

Gz=generator(z_placeholder,batch_size,z_dimensions)
Dx=discriminator(x_placeholder)
Dg=discriminator(Gz,reuse_variables=True)
#Gz=假圖、Dx=真圖的分數、Dg=假圖的分數

#Get the variables for different network
tvars=tf.trainable_variables()
d_vars=[var for var in tvars if 'd_' in var.name]
g_vars=[var for var in tvars if 'g_' in var.name]

#tf.ones_like(x)和tf.zeros_like(x) 分別會產生x個1和0
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.zeros_like(Dg)))
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)))
#Train the discrimiinator
d_trainer_fake=tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake,var_list=d_vars)
d_trainer_real=tf.train.AdamOptimizer(0.0003).minimize(d_loss_real,var_list=d_vars)
#Train the generator
g_trainer=tf.train.AdamOptimizer(0.0001).minimize(g_loss,var_list=g_vars)

#Start Training session
saver=tf.train.Saver()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
tf.get_variable_scope().reuse_variables()
#pre-train discriminator
#for i in range(300):
#	z_batch=np.random.normal(0,1,size=[batch_size,z_dimensions])
#	real_image_batch=mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
#	_,__,dLossReal,dLossFake=sess.run([d_trainer_real,d_trainer_fake,d_loss_real,d_loss_fake],feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
#	if i %100==0:
#		print("dLossReal:", dLossReal,"dLossFake:", dLossFake)

#Train generator and discriminator together

for i in range(1000):
	real_image_batch=mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	z_batch=np.random.normal(0,1,size=[batch_size,z_dimensions])
	#Train discriminator on both real and fake images
	_,__,dLossReal,dLossFake=sess.run([d_trainer_real,d_trainer_fake,d_loss_real,d_loss_fake],feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
	#Train generator
	#z_batch=np.random.normal(0,1,size=[batch_size,z_dimensions])
	_=sess.run(g_trainer,feed_dict={z_placeholder:z_batch})
	img=sess.run(Gz,feed_dict={z_placeholder:z_batch})
	if i %10==0:
		print("dLossReal:", dLossReal,"dLossFake:", dLossFake)

for i in range(10):
	generated=img[i].reshape([28,28])
	real=real_image_batch[i].reshape([28,28])
	file="test"+str(i)
	file1="real"+str(i)
	plt.imshow(generated,cmap="gray")
	plt.savefig("C:/Users/user/Desktop/graph/"+file)
	plt.imshow(real,cmap="gray")
	plt.savefig("C:/Users/user/Desktop/graph/"+file1)




#For setting TensorBoard
#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
#tf.get_variable_scope().reuse_variables()

#tf.summary.scalar('generator_loss',g_loss)
#tf.summary.scalar('Discriminator_loss_real',d_loss_real)
#tf.summary.scalar('Discriminator_loss_fake',d_loss_fake)

#images_for_tensorboard=generator(z_placeholder,batch_size,z_dimensions)
#tf.summary.image('generated_image',images_for_tensorboard,5)
#merged=tf.summary.merge_all()
#train_writer=tf.summary.FileWriter('C:/Users/user/Downloads/pythonCode/GANtf',sess.graph)
#tensorboard --logdir=C:/Users/user/Downloads/pythoncode/GANtf --host=127.0.0.1





#generated_image_output=generator(z_placeholder,10,z_dimensions)

#z_batch=np.random.normal(0,1,[1,z_dimensions])

#with tf.Session() as sess:
#	sess.run(tf.global_variables_initializer())
#	generated_image=sess.run(generated_image_output,feed_dict={z_placeholder:z_batch})
#	generated_image=generated_image.reshape([28,28])
#	plt.imshow(generated_image,cmap='Greys')
#	plt.show()