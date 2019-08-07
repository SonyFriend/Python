import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img=np.array(Image.open('C:/Users/user/Desktop/cat.jpg')) #打開圖片並轉成數字矩陣
#plt.figure("cat")
#plt.imshow(img)
#plt.axis('off')
#plt.show()
#print(img.shape) #shape=194,259,3 

#隨機生成3000個椒鹽噪聲
row,col,dim=img.shape
for i in range(3000):
	x=np.random.randint(0,row)
	y=np.random.randint(0,col)
	img[x,y,:]=255

#將圖像二值化,像素值大於128轉為1,反之為0
#img1=np.array(Image.open('C:/Users/user/Desktop/cat.jpg').convert('L')) #打開圖片並轉成數字矩陣 convert('L')轉呈灰色圖片
#rows,cols=img1.shape
#for i in range(rows):
#	for j in range(cols):
#		if (img1[i,j]<=128):
#			img1[i,j]=0
#		else:
#			img1[i,j]=1

