import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets

iris=datasets.load_iris()
x=pd.DataFrame(iris['data'],columns=iris['feature_names'])
y=pd.DataFrame(iris['target'],columns=['target'])
data=pd.concat([x,y],axis=1)
iris_data=data[['sepal length (cm)','petal length (cm)','target']]
iris_data=iris_data[iris_data['target'].isin([0,1])]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(iris_data[['sepal length (cm)','petal length (cm)']],iris_data[['target']],test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)

from sklearn.svm import SVC
svm=SVC(kernel='linear',probability=True)
svm.fit(x_train_std,y_train['target'].values)
#from sklearn import metrics
#print(metrics.classification_report(y_test,svm.predict(x_test_std)))
#print(metrics.confusion_matrix(y_test,svm.predict(x_test_std)))
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_train_std,y_train['target'].values,clf=svm)
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()