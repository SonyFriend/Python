import numpy as np
import pandas as pd
import csv
import keras
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import Adam
import jieba.posseg as pseg
#train=pd.read_csv("C:/Users/user/Downloads/pythonCode/FakeNews/train.csv",index_col=0)
#title=['title1_zh','title2_zh','label']
#train=train.loc[:,title]
######################文字切割############
def jieba_tokenizer(text):
	words=pseg.cut(text)
	return ' '.join([word for word, flag in words if flag !='x'])#'x' 移除標點符號

#train['title1_tokenized']=train.loc[:,'title1_zh'].apply(jieba_tokenizer)
#t=train.loc[:,'title2_zh'].astype(str)
#train['title2_tokenized']=t.apply(jieba_tokenizer)

############切割完建立新檔案#############
#with open('C:/Users/user/Downloads/pythonCode/FakeNews/newtrain.csv','w',newline='',encoding="utf-8") as f:
#	writer=csv.writer(f)
#	writer.writerow(train.columns.values.tolist())
#	for i in range(0,train.shape[0]):
#		writer.writerow(train.iloc[i,:])

train=pd.read_csv("C:/Users/user/Downloads/pythonCode/FakeNews/newtrain.csv",index_col=0,encoding="utf-8")
train=train.fillna('')
#print(corpus.shape)
#print(pd.DataFrame(corpus.iloc[:5],columns=['title']))
##########建立辭典#################
max_num_words=10000
#tokenizer 將文字轉成一系列的詞彙  max_num_words辭典所能容納的詞數
tokenizer=keras.preprocessing.text.Tokenizer(num_words=max_num_words)
corpus_x1=train.title1_tokenized
corpus_x2=train.title2_tokenized

corpus=pd.concat([corpus_x1,corpus_x2])
#print(corpus.shape)
tokenizer.fit_on_texts(corpus)

x1_train=tokenizer.texts_to_sequences(corpus_x1)
x2_train=tokenizer.texts_to_sequences(corpus_x2)

#print(x1_train[:1])
###########tokenizer.index_word可將索引數字對應回文字###########
#for seq in x1_train[:1]:
#	print([tokenizer.index_word[idx] for idx in seq])
#max_seq_len=max([len(seq) for seq in x1_train]) #最長的序列有61個詞彙
#print(max_seq_len)

max_sequence_length=20#為了讓所有序列長度一致 長度超過20序列尾巴會被刪掉 不足20的詞彙前面會補0
x1_train=keras.preprocessing.sequence.pad_sequences(x1_train,maxlen=max_sequence_length)
x2_train=keras.preprocessing.sequence.pad_sequences(x2_train,maxlen=max_sequence_length)

label_to_index={'unrelated':0,'agreed':1,'disagreed':2}
y_train=train.label.apply(lambda x:label_to_index[x])
y_train=np.asarray(y_train).astype('float32')

y_train=keras.utils.to_categorical(y_train)
x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1_train,x2_train,y_train,test_size=0.1,random_state=0)

#print(x1_train.shape)#shape (288496,20)
#print(x2_train.shape)#shape(288496,20)
# 一個詞向量的維度
num_embedding_dim= 256
# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128

from keras import Input
from keras.layers import Embedding,LSTM,concatenate,Dense
from keras.models import Model
#詞嵌入層
#經過詞嵌入層轉換,兩個新聞標題都變成一個詞向量的序列
#每個詞向量的維度=256
top_input=Input(shape=(max_sequence_length, ),dtype='int32')
bm_input=Input(shape=(max_sequence_length, ),dtype='int32')

embedding_layer=Embedding(max_num_words,num_embedding_dim)
top_embedded=embedding_layer(top_input)
bm_embedded=embedding_layer(bm_input)

#LSTM層
#兩個新聞標題經過此層後,為一個128維度向量

shared_lstm=LSTM(NUM_LSTM_UNITS)
top_output=shared_lstm(top_embedded)
bm_output=shared_lstm(bm_embedded)
#串接層將兩個新聞標題的結果串接成單一向量
#方便跟全連接層相連
merged=concatenate([top_output,bm_output],axis=-1)
#全連接層搭配softmax activation
#可以回傳3個成對標題
#屬於各類別的機率

dense=Dense(units=3,activation='softmax')
predictions=dense(merged)
model=Model(inputs=[top_input,bm_input],outputs=predictions)
from keras.layers import Dropout
from keras.optimizers import Adam
lr=0.001
adam=Adam(lr)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])



batch_size=512
num_epochs=1
history=model.fit(x=[x1_train,x2_train],y=y_train,batch_size=batch_size,epochs=num_epochs,validation_data=([x1_test,x2_test],y_test),shuffle=True)

#test=pd.read_csv("C:/Users/user/Downloads/pythonCode/FakeNews/test.csv",index_col=0,encoding="utf-8")
#title=['title1_zh','title2_zh']
#test=test.loc[:,title]

#test['title1_tokenized']=test.loc[:,'title1_zh'].apply(jieba_tokenizer)
#t_test=test.loc[:,'title2_zh'].astype(str)
#test['title2_tokenized']=t_test.apply(jieba_tokenizer)
############切割完建立新檔案#############
#with open('C:/Users/user/Downloads/pythonCode/FakeNews/newtest.csv','w',newline='',encoding="utf-8") as f:
#	writer=csv.writer(f)
#	writer.writerow(test.columns.values.tolist())
#	for i in range(0,test.shape[0]):
#		writer.writerow(test.iloc[i,:])

test=pd.read_csv("C:/Users/user/Downloads/pythonCode/FakeNews/newtest.csv",index_col=0,encoding="utf-8")
test=test.fillna('')

corpus_x1_test=test.title1_tokenized
corpus_x2_test=test.title2_tokenized

corpus_test=pd.concat([corpus_x1_test,corpus_x2_test])

x1_test=tokenizer.texts_to_sequences(corpus_x1_test)
x2_test=tokenizer.texts_to_sequences(corpus_x2_test)

x1_test=keras.preprocessing.sequence.pad_sequences(x1_test,maxlen=max_sequence_length)
x2_test=keras.preprocessing.sequence.pad_sequences(x2_test,maxlen=max_sequence_length)
predictions = model.predict([x1_test, x2_test])

index_to_label={v:k for k,v in label_to_index.items()}
test['Category']=[index_to_label[idx] for idx in np.argmax(predictions,axis=1)]
submission=test.loc[:,'Category'].reset_index()
submission.columns=['Id','Category']
print(submission.head())
