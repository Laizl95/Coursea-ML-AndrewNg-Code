import matplotlib as mpl
import matplotlib.pyplot as plt  
import numpy as np

def f(a,x):
	return np.dot(a,x)
a=np.array([2.,1.])
data=np.loadtxt("C:/Users/lai/Desktop/Coursea机器学习作业/ex1data1.txt", delimiter=',')
#print(data[:,0])print(data.shape)
x1=np.asarray(data[:,0])
x0=np.ones_like(x1)
x=np.vstack((x0,x1)).T
y=np.asarray(data[:,1])
#print(f(np.array([1,2]),np.array([2,4])))
#x1=np.array([3,2])x2=np.array([[1,6.1101],[1,5.5277],[1,8.5186]])print(x1*x2)print(np.sum(np.sum(a*x,axis=1)-y))print((np.sum(a*x,axis=1)-y))print(x[:,1])print((np.sum(a*x,axis=1)-y)*x[:,1])
t=0.001
#print(sum*sum)print(x[:,1])print(sum*x[:,1])print((np.sum(a*x,axis=1)-y)*2)
sum=np.sum(a*x,axis=1)-y
error=(1./(2*x.shape[0]))*np.sum(sum*sum)
#print(x)print(y)print(error)
pre=error+1
#print(sum,' ',sum*sum)
while pre-error>=0.001:
	for i in range(len(a)):
		#print('lkjujuy',(1./x.shape[0])*np.sum(sum*x[:,i]))
		a[i]=a[i]-t*(1./x.shape[0])*np.sum(sum*x[:,i])
	sum=np.sum(a*x,axis=1)-y
	print(error)
	pre=error
	error=(1./(2*x.shape[0]))*np.sum(sum*sum)
	print(error)
print(a)
t=np.linspace(0,20,70)
y1=t*(np.full(t.shape,a[1]))+(np.full(t.shape,a[0]))
#print(y,t)
plt.plot(x1,y,'bo')
plt.plot(t,y1)
plt.show()