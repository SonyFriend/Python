import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import csv
df=open('C:/Users/user/Downloads/pythonCode/movie_us_20920.csv')
movie=pd.read_csv(df) 

row=[i for i in range(0,len(movie['budget'])) if movie.iat[i,2]==0]
movie=movie.drop(index=row)
row=None
row=[i for i in range(0,len(movie['revenue'])) if movie.iat[i,15]==0]
movie=movie.drop(index=movie.index[row])

#print(movie.columns.values.tolist()[0:2])
with open('C:/Users/user/Downloads/pythonCode/movie_4381.csv','w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow(movie.columns.values.tolist())
	for i in range(0,len(movie)):
		writer.writerow(movie.iloc[i,:])
