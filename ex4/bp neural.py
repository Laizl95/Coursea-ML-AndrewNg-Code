import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def plot_images(X):
	sample_id=np.random.choice(np.arange(X.shape[0]),100)
	images=X[sample_id,:]
	#print(images.shape)
	fig,ax=plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize=(8,8))
	for r in range(10):
		for c in range(10):
			ax[r,c].matshow(images[r*10+c].reshape(20,20).T,cmap=plt.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
	#plt.show()

def load_data(path):
	data=scio.loadmat(path)
	return data['X'],data['y']
def load_weight(path):
	weight=scio.loadmat(path)
	return weight['Theta1'],weight['Theta2']
def sigmoid(z):
	return 1./(1+np.exp(-z))
def cost(theta,x,y):
	return -1*np.mean(y*np.log(sigmoid(theta,x))+(1-y)*np.log(1-sigmoid(theta,x)))
def gradient(theta,x,y):
	return (1./y.shape[0])*np.dot(sigmoid(theta,x)-y,x)
def process_data():
	path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex4/ex4data1.mat'
	X,y=load_data(path)
	#print(X.shape,' ',y.shape)
	plot_images(X)
	X=np.insert(X,0,np.ones_like(X.shape[0]),axis=1)
	#print(X.shape)
	y_mat=[]
	y=np.reshape(y,y.shape[0])
	for i in range(1,11):
		y_mat.append((y==i).astype(int))
	#print(y_mat[:2])第0行到第一行print(y_mat[-1:],' ',y_mat[1,:],' ',y_mat[-1])
	#y_mat=y_mat[-1:]+y_mat[:-1]
	y=np.array(y_mat)
	#print(y)
	return X,y
def feed_forward(X,theta1,theta2):
	z2=np.dot(X,theta1.T)
	a2=sigmoid(z2)
	a2=np.insert(a2,0,np.ones_like(a2.shape[1]),axis=1)
	print(a2.shape)
	z3=np.dot(a2,theta2.T)
	a3=sigmoid(z3)
	return z2,a2,z3,a3
def cal_cost(y,h):
	print(h.shape,' ',y.shape)
	cost_mat=-y.T*np.log(h)-(1-y.T)*np.log(1-h)
	print(cost_mat.shape)
	cost=np.sum(cost_mat)/5000
	print(cost)
	regular_term=1/10000*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
	print(cost+regular_term)


def deserialize(seq):
#     """into ndarray of (25, 401), (10, 26)"""
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)
a=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
print(a[2:3*3])
X,y=process_data()
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex4/ex4weights.mat'
theta1,theta2=load_weight(path)
print(theta1.shape,' ',theta2.shape)
z2,a2,z3,a3=feed_forward(X,theta1,theta2)
cal_cost(y,a3)