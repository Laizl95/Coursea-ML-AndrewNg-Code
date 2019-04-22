import matplotlib as mpl
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
def get(x):
	return (x-np.mean(x))/np.std(x)

def h(theta,x):
	return 1./(1+np.exp(-np.sum(theta*x,axis=1)))
thetas=np.array([0.,0.,0.])
data=np.loadtxt("C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex2/ex2data1.txt", delimiter=',')
x0=np.ones_like(data[:,0])
data[:,0]=get(data[:,0])
data[:,1]=get(data[:,1])
x=np.vstack((x0,data[:,0],data[:,1])).T
y=data[:,2]
print(thetas*x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
for i in range(len(y)):
	if y[i]==0.:
		plt.plot(x[i,1],x[i,2],'bo')
	else:
		plt.plot(x[i,1],x[i,2],'gx')
a=0.01
cnt=0
m=x.shape[0]
#print(np.exp(-np.sum(theta*x,axis=1)))
#print(h(theta,x))
while cnt<=100:
	t=h(thetas,x)
	s=t-y
	for i in range(len(thetas)):
		#print('sum',np.sum(s*x[:,i]))
		thetas[i]=thetas[i]-a*np.sum(s*x[:,i])
	t=h(thetas,x)
	cost=-(1./m)*np.sum(y*np.log(t)+(1-y)*np.log(1-t))
	cnt+=1
print(thetas)
t=np.linspace(np.min(data[:,0]),np.max(data[:,1]),80)
y=-(thetas[0]+thetas[1]*t)/(thetas[2])
plt.plot(t,y)
plt.show()