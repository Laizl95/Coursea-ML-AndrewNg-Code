import matplotlib as mpl
import matplotlib.pyplot as plt  
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def f(a,x):
	return np.dot(a,x)
a=np.array([2.,1.,1.])
data=np.loadtxt("C:/Users/lai/Desktop/Coursea机器学习作业/ex1data2.txt", delimiter=',')
x1=np.asarray(data[:,0])
x0=np.ones_like(x1)
max_x1=np.full(x1.shape,np.max(x1))
min_x1=np.full(x1.shape,np.min(x1))
mean_x1=np.full(x1.shape,np.mean(x1))
x2=np.asarray(data[:,1])
max_x2=np.full(x2.shape,np.max(x2))
min_x2=np.full(x2.shape,np.min(x2))
mean_x2=np.full(x2.shape,np.mean(x2))
x=np.vstack((x0,x1,x2)).T
y=np.asarray(data[:,2])
mean_y=np.full(y.shape,np.mean(y))
max_y=np.full(y.shape,np.max(y))
min_y=np.full(y.shape,np.min(y))
x[:,1]=(x[:,1]-mean_x1)/(max_x1-min_x1)
x[:,2]=(x[:,2]-mean_x2)/(max_x2-min_x2)
y=(y-mean_y)/(max_y-min_y)
#print(max_x1,' ',min_x1,' ',mean_x1,' ',max_x2,' ',min_x2,' ',mean_x2,' ',max_y,' ',min_y,' ',mean_y)
print(x,'\n',y)
#print(f(np.array([1,2]),np.array([2,4])))
#x1=np.array([3,2])x2=np.array([[1,6.1101],[1,5.5277],[1,8.5186]])print(x1*x2)print(np.sum(np.sum(a*x,axis=1)-y))print((np.sum(a*x,axis=1)-y))print(x[:,1])print((np.sum(a*x,axis=1)-y)*x[:,1])
t=1
#print(sum*sum)print(x[:,1])print(sum*x[:,1])print((np.sum(a*x,axis=1)-y)*2)
sum=np.sum(a*x,axis=1)-y
error=(1./(2*x.shape[0]))*np.sum(sum*sum)
pre=error+1
#print(sum,' ',sum*sum)
cnt=0
while cnt<=60:
	for i in range(len(a)):
		a[i]=a[i]-t*(1./x.shape[0])*np.sum(sum*x[:,i])
	sum=np.sum(a*x,axis=1)-y
	pre=error
	error=(1./(2*x.shape[0]))*np.sum(sum*sum)
	print(pre-error)
	cnt+=1
print(a)
t1=np.linspace(-0.5,0.5,100)
t2=np.linspace(-0.5,0.5,100)
T1,T2=np.meshgrid(t1,t2)
y1=T2*(np.full(t2.shape,a[2]))+T1*(np.full(t1.shape,a[1]))+(np.full(t1.shape,a[0]))
#print(y,t)
fig=plt.figure()
ax=fig.add_subplot('111',projection='3d')
ax.scatter(x[:,1],x[:,2],y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.plot_surface(T1,T2,y1)
ax.legend()
plt.show()