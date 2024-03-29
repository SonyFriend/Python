import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST.data',one_hot='True') #one_hot表示用非0即1的數字保存圖片
#hyperparameter
lr=0.001 #learning rate
training_iters=1000000#循環次數
batch_size=128#每個批次餵進RNN的數據大小

n_inputs=28 #MNIST data input(img shape:28*28) 每次iinput一列的pixel 有28個
n_steps=28 #time step 有28列的pixel
n_hidden_units=128 #隱藏層神經元個數
n_classes=10 #MNIST classes(0-9 digits)

#tf Graph input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

#Define weights
weights={#(28,128)
'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
#(128,10)
'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))}
biases={#(128,)
'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
#(10,)
'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))}

def RNN(X,weights,biases):
	#hidden layer for input to cell
	#X(128 batch,28 steps,28inputs)->(128*28,28 inputs)
	X=tf.reshape(X,[-1,n_inputs])
	#X_in=(128 batch*28 steps,128 hidden)
	X_in=tf.matmul(X,weights['in'])+biases['in']
	#X_in=(128 batch,28 steps,128 hidden)
	X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])
	#cell
	lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
	#lstm cell is divided into two parts(c_state,m_state) c_state 主線 m_state 分線
	init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
	outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False) #time_major看time steps 是否為主要維度 X_in=(128 batch,"28 steps",128 hidden)
	#hidden layer for output as the final results
	results=tf.matmul(states[1],weights['out'])+biases['out']
	return results

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
sess=tf.Session()
sess.run(tf.global_variables_initializer())

step=0
while step*batch_size <training_iters:
	batch_xs,batch_ys=mnist.train.next_batch(batch_size)
	print(batch_xs.shape)
	batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
	sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys,})
	if step %20==0:
		#print(sess.run(tf.argmax(pred,1),feed_dict={x:batch_xs})) 預測結果
		#print(sess.run(tf.argmax(y,1),feed_dict={y:batch_ys})) 實際結果
		print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,}))
	step+=1