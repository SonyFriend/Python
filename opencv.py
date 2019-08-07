import cv2 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
plt.imshow(img1[5])
plt.show()
