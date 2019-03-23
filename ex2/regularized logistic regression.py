import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def f(x,y,theta):
	final_x=[]
	for i in range(7):
		for j in range(7):
			if i+j<=6:
				final_x.append((x**i)*(y**j))
				sum=0
	for i in range(len(final_x)):
		sum+=theta[i]*final_x[i]
	return sum

def h(theta,x):
	return 1./(1+np.exp(-np.sum(theta*x,axis=1)))
def normal(x):
	return (x-np.mean(x))/(np.max(x)-np.min(x))
print(2**10)
data=np.loadtxt("C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex2/ex2data2.txt", delimiter=',')
x=data[:,0]
x1=data[:,0]
x2=data[:,1]
for i in range(7):
	for j in range(7):
		if i+j<=6:
			x=np.row_stack((x,np.power(x1,i)*np.power(x2,j)))
x=np.delete(x,0,0)
x=x.T
print(x.shape[1])
for i in range(x.shape[1]):
	if i!=0:
		x[:,i]=normal(x[:,i])
y=data[:,2]
for i in range(len(y)):
	if y[i]==0.:
		plt.plot(x[i,1],x[i,7],'bo')
	else:
		plt.plot(x[i,1],x[i,7],'gx')

a=0.259
cnt=0
thetas=np.zeros_like(x[0,:])
thetas=np.array(thetas)
#print(thetas*x)
#print(x)
m=x.shape[0]
#print(np.exp(-np.sum(theta*x,axis=1)))
#print(h(theta,x))
while cnt<=3000:
	t=h(thetas,x)
	s=t-y
	for i in range(len(thetas)):
		#print('sum',np.sum(s*x[:,i]))
		if i==0:
			thetas[i]=thetas[i]-a*np.sum(s*x[:,i])
		else:
			thetas[i]=thetas[i]-a*np.sum(s*x[:,i])-47/m*thetas[i]
	t=h(thetas,x)
	cost=-(1./m)*np.sum(y*np.log(t)+(1-y)*np.log(1-t))+48.7/m*np.sum(thetas*thetas)
	
	#print(cost)
	cnt+=1
#print(thetas)
t1=np.linspace(np.min(x[:,1]),np.max(x[:,1]),80)
t2=np.linspace(np.min(x[:,7]),np.max(x[:,7]),80)
T1,T2=np.meshgrid(t1,t2)
plt.contour(T1,T2,f(T1,T2,thetas),0)
plt.show()

