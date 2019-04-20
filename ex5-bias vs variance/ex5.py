import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def load_data(path):
	data=scio.loadmat(path)
	#print(data)
	return data['X'],data['y'],data['Xtest'],data['ytest'],data['Xval'],data['yval']

def linear_regression(theta,x):
	return np.dot(x,theta)

def regression_gradient(y,h,X,regular,theta):
	m=X.shape[0]
	deleta=1./m*np.dot((h-y).T,X)
	if(regular==True):
		deleta[1:]+=1./m*theta[1:]
	return deleta
def cost():
	m=X.shape[0]
	J=np.sum((h-y)*(h-y))
	if(regular==True):
		J+=np.sum((theta*theta)[:,1:])
	return 1/(2*m)*J
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex5/ex5data1'
X,y,Xtest,ytest,Xval,yval=load_data(path)
X=np.insert(X,0,np.ones_like(X.shape[0]),axis=1)
Xval=np.insert(Xval,0,np.ones_like(Xval.shape[0]),axis=1)
Xtest=np.insert(Xtest,0,np.ones_like(Xtest.shape[0]),axis=1)
print(X)
print(y.shape)
plt.scatter(X[:,1],y)
plt.show()
a=np.array([[1,2,3],[4,5,6],[7,8,9],[11,11,11]])
print(a[:,1:])
print(np.sum(a[:,1:],axis=0))

