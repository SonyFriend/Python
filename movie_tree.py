import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv

df = open('C:/Users/user/Downloads/統諮FINAL/movie_4381.csv')
#with open('C:\\Users\\user\\Downloads\\統諮FINAL\\movie_4381.csv',"r") as data:
#	movie=csv.reader(data)
movie=pd.read_csv(df)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(movie[['runtime','revenue','budget','popularity']],movie[['vote_average']],test_size=0.3,random_state=0)
tree=DecisionTreeRegressor(criterion='mse',max_depth=1,random_state=0)
tree=tree.fit(x_train,y_train)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus 

with open("tree.dot", 'w') as f:
 f = export_graphviz(tree, out_file=f)

dot_data=export_graphviz(tree,out_file=None,feature_names=['runtime','revenue','budget','popularity'],filled=True,rounded=True,special_characters=True)

y=y_test['vote_average']
predict=tree.predict(x_test)

graph=pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("tree.pdf") 
from sklearn import metrics

print(np.mean(abs(np.multiply(np.array(y_test.T-predict),np.array(1/y_test)))))