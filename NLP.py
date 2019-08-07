import numpy as np
import pandas as pd 
import re #library to clean data
import nltk #Natural Language tool kit 
#nltk.download('stopwords') 
from nltk.corpus import stopwords #to remove stopword
from nltk.stem.porter import PorterStemmer #提取詞幹
corpus=[]#Initialize empty array to append clean text
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
#print(dataset[0:3]) dataset 1000 row 2 col
#1000(reviews) rows to clean
for i in range(1000):
	reviews=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #[^a-zA-Z] 非a-z或A-Z的範圍 取代成空白字元
	reviews=reviews.lower() #convert to lower case
	reviews=reviews.split() #split to array(default delimiter is " ")
	ps=PorterStemmer() #creating porterStemmer object to take main stem of each word
	reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))] #loop for stemming each word  in string array at ith row
	reviews=' '.join(reviews)#rejoin all string array element to create back into a string 字符間加入空白
	corpus.append(reviews)#append each string to create array of clean text

from sklearn.feature_extraction.text import CountVectorizer#Countvectorizer是一個文字特徵提取方法，對於每一個訓練文字只考慮每種詞彙在該訓練文字中出現的頻率
cv=CountVectorizer(max_features=1500)#to extract max 1500 feature　只取前1500個做為關鍵詞集
X=cv.fit_transform(corpus).toarray()#fit_transform函式計算各個詞語出現的次數 toarray()可看到詞頻矩陣的結果。
y=dataset.iloc[:,1].values#y contain answer if review is postive or negative
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=1000,criterion='entropy')#n_estimators cab be said as number of trees
model.fit(X_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
acc=(cm[0,0]+cm[1,1])/sum(sum(cm))
print('accuracy',acc)
print(cm)