import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1) #設定CPU的種子
np.random.seed(1)

Batch_size=64
lr_G=0.0001 #learning rate for generator
lr_D=0.0001 #learning rate for discriminator
N_ideas=5 #number of ideas for generating an art work(generator)
art_components=15 #total point G can draw in the curve
#np.vstack 沿著直的方向將矩陣疊起來 np.linspace 生成等差數列
paint_points=np.vstack([np.linspace(-1,1,art_components) for _ in range(Batch_size)])

def artist_works():
	a=np.random.uniform(1,2,size=Batch_size)[:,np.newaxis] #np.newaxis 將1xbatch_size矩陣轉置
	paintings=a*np.power(paint_points,2)+(a-1)
	paintings=torch.from_numpy(paintings).float()
	return paintings
#Generator
G=nn.Sequential(nn.Linear(N_ideas,128),nn.ReLU(),nn.Linear(128,art_components))#nn.Linear(x,y),x和y分別為X和Y的維度,用來處理Y=Wx+b之類的線性函數
#Discriminator
#nn.Linear(art_components,128) receive art work either from the famous artist or a newbie like G
#nn.Sigmoid() tell the probability that art work is made by artist
D=nn.Sequential(nn.Linear(art_components,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())

opt_D=torch.optim.Adam(D.parameters(),lr=lr_D)
opt_G=torch.optim.Adam(G.parameters(),lr=lr_G)

plt.ion()

for step in range(10000):
	artist_paintings=artist_works() #real painting from artist
	G_ideas=torch.randn(Batch_size,N_ideas) #random ideas
	G_paintings=G(G_ideas) #fake painting from G

	prob_artist0=D(artist_paintings) #D要增加判斷真畫的機率
	prob_artist1=D(G_paintings)#D要減少由G畫出是真畫的機率

	D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1. -prob_artist1))
	G_loss=torch.mean(torch.log(1. -prob_artist1))

	opt_D.zero_grad()
	D_loss.backward(retain_graph=True) #reusung computational graph
	opt_D.step()

	opt_G.zero_grad()
	G_loss.backward()
	opt_G.step()

	if step%50==0:
		plt.cla() #clear axis
		plt.plot(paint_points[0],G_paintings.data.numpy()[0],c='#4AD631',lw=3,label='Generated painting')
		plt.plot(paint_points[0],2*np.power(paint_points[0],2)+1,c='#74BCFF',lw=3,label='upper bound')
		plt.plot(paint_points[0],1*np.power(paint_points[0],2)+0,c='#FF9359',lw=3,label='lower bound')
		plt.text(-.5,2.3,'D accuracy=%.2f (0.5 for D to converge' % prob_artist0.data.numpy().mean(),fontdict={'size':13})
		plt.text(-.5,2,'D score=%.2f (-1.38 for G to converge' % -D_loss.data.numpy(),fontdict={'size':13})
		plt.ylim((0,3));plt.legend(loc='upper right',fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()
