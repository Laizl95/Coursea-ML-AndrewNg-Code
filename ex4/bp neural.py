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
	#print(y_mat[:2])第0行到第一行print(y_mat[-1:],' ',y_mat[1,:],' ',y_mat[-1]) y_mat[:3*2]0-5
	#y_mat=y_mat[-1:]+y_mat[:-1]
	y=np.array(y_mat)
	#print(y)
	return X,y
def feed_forward(X,theta1,theta2):
	z2=np.dot(X,theta1.T)
	a2=sigmoid(z2)
	a2=np.insert(a2,0,np.ones_like(a2.shape[1]),axis=1)
	#print(a2.shape)
	z3=np.dot(a2,theta2.T)
	a3=sigmoid(z3)
	return z2,a2,z3,a3
def cal_cost(y,h):
	#print(h.shape,' ',y.shape)
	cost_mat=-y.T*np.log(h)-(1-y.T)*np.log(1-h)
	#print(cost_mat.shape)
	cost=np.sum(cost_mat)/5000
	print('cost=',cost)
	regular_term=1/10000*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
	print('regual_cost=',cost+regular_term)

def sigmoid_gradient(z):
	return sigmoid(z)*(1-sigmoid(z))

def gradient(y,theta1,theta2,X):
	m=X.shape[0]
	y_n=y.T
	delta2=np.zeros(theta2.shape)
	delta1=np.zeros(theta1.shape)
	z2,a2,z3,a3=feed_forward(X,theta1,theta2)
	for i in range(m):
		a3i=a3[i,:]
		yi=y_n[i,:]
		a2i=a2[i,:]
		gi=sigmoid_gradient(a3i)*(yi-a3i)
		#print(gi.shape)
		delta2+=np.dot(np.matrix(gi).T,np.matrix(a2i))
		#print(delta2.shape)
		ei=np.multiply(np.matrix(sigmoid_gradient(a2i)).T,np.dot(theta2.T,np.matrix(gi).T))
		#print(ei.shape)
		delta1+=np.dot(ei[1:],np.matrix(X[i,:]))
		#print(delta1.shape,' ',delta2.shape)
	t1=theta1
	t2=theta2
	t1[:,0]=t2[:,0]=0
	return (delta1+t1)/m,(delta2+t2)/m,a3


X,y=process_data()
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex4/ex4weights.mat'
theta1,theta2=load_weight(path)
print('X',X.shape)
print(theta1.shape,' ',theta2.shape)


z2,a2,z3,a3=feed_forward(X,theta1,theta2)
cal_cost(y,a3)
print(z2.shape,' ',a2.shape,' ',z3.shape,' ',a3.shape,' ',y.shape)


init_theta1=np.random.uniform(-0.12,0.12,theta1.shape)
init_theta2=np.random.uniform(-0.12,0.12,theta2.shape)
t=0
while(t<=150):
	d1,d2,h=gradient(y,init_theta1,init_theta2,X)
	init_theta1+=d1
	init_theta2+=d2
	t+=1

	
z2,a2,z3,a3=feed_forward(X,init_theta1,init_theta2)
cal_cost(y,a3)
print(a3.shape)
cnt_y=(np.array([a3[:,i] for i in range(10)])>=0.5).astype(int)
#cnt_y0=(p[:,0]>=0.5).astype(int)print(cnt_y.shape)
print(cnt_y.shape)
print(y.shape)
print(np.mean([cnt_y[i,:]==y[i,:] for i in range(10)],axis=1))

def plot_hidden(theta):

	fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(5,5))
	theta=theta[:,1:]
	for r in range(5):
		for c in range(5):
			ax[r,c].matshow(theta[r*5+c].reshape(20,20).T,cmap=plt.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
