import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def plot_an_image(image):
    fig,ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=plt.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

def plot_images(X):
	size=int(np.sqrt(X.shape[1]))
	sample_id=np.random.choice(np.arange(X.shape[0]),100)
	sample_images=X[sample_id,:]
	#print(X[4,:].shape)print(sample_images[2].shape)print(sample_images.shape)
	fig,ax=plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(8,8))
	for r in range(10):
		for c in range(10):
			ax[r,c].matshow(sample_images[r*10+c].reshape((size,size)),cmap=plt.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))

def load_data(path):
	data=scio.loadmat(path)
	y=data.get('y')
	y=y.reshape(y.shape[0])
	X=data.get('X')
	X=np.array([im.reshape((20, 20)).T for im in X])
	X=np.array([im.reshape(400) for im in X])
	return X,y
#a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
#print(a)
#print(np.array([im.reshape((2,2)) for im in a]))
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex3/ex3data1.mat'
X,y=load_data(path)
pick_one=np.random.randint(0,5000)
#plot_an_image(X[pick_one,:])
plot_images(X)
#plt.show()
def sigmoid(z):
	return 1;
raw_X,raw_y=load_data(path)
X=np.insert(raw_X,0,values=np.ones_like(raw_X.shape[0]),axis=1)
y_mat=[]
raw_y=np.array([1,2,3])
for k in range(1,11):
	y_mat.append((raw_y==k).astype(int))
#print(y_mat)	
#print(y_mat[-1:])
y_mat=[y_mat[-1]]+y_mat[:-1]
#print(y_mat[1:])
y=np.array(y_mat)
#print(y)

import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def plot_an_image(image):
    fig,ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=plt.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

def plot_images(X):
	size=int(np.sqrt(X.shape[1]))
	sample_id=np.random.choice(np.arange(X.shape[0]),100)
	sample_images=X[sample_id,:]
	'''print(X[4,:].shape)
	print(sample_images[2].shape)
	print(sample_images.shape)'''
	fig,ax=plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(8,8))
	for r in range(10):
		for c in range(10):
			ax[r,c].matshow(sample_images[r*10+c].reshape((size,size)),cmap=plt.cm.binary)
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))

def load_data(path):
	data=scio.loadmat(path)
	y=data.get('y')
	y=y.reshape(y.shape[0])
	X=data.get('X')
	#X=np.array([im.reshape((20, 20)).T for im in X])
	#X=np.array([im.reshape(400) for im in X])
	return X,y
#a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
#print(a)
#print(np.array([im.reshape((2,2)) for im in a]))
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex3/ex3data1.mat'
X,y=load_data(path)
pick_one=np.random.randint(0,5000)
#plot_an_image(X[pick_one,:])
plot_images(X)
#plt.show()
def sigmoid(theta,x):
	return 1./(1+np.exp(-np.dot(x,theta)))
def cost(theta,x,y):
	return -1*np.mean(y*np.log(sigmoid(theta,x))+(1-y)*np.log(1-sigmoid(theta,x)))
def gradient(theta,x,y):
	return (1./y.shape[0])*np.dot(sigmoid(theta,x)-y,x)
def logistic_regression(x,y):
	theta=np.zeros(x.shape[1])
	#print(theta.shape,' ',x.shape,' ',y.shape)
	cnt=1
	a=1
	while cnt<=100:
		theta=theta-a*gradient(theta,x,y)
		#print(cost(theta, x, y))
		cost(theta, x, y)
		cnt+=1
	return theta
def predict(theta,x):
	p=sigmoid(theta, x)
	return (p>=0.5).astype(int)
def predict_k(theta,x):
	p=sigmoid(theta, x)
	#print(p[:,0])
	cnt_y=(np.array([p[:,i] for i in range(10)])>=0.5).astype(int)
	#cnt_y0=(p[:,0]>=0.5).astype(int)print(cnt_y.shape)
	print(np.mean([cnt_y[i,:]==y[i] for i in range(10)],axis=1))
raw_X,raw_y=load_data(path)
X=np.insert(raw_X,0,values=np.ones_like(raw_X.shape[0]),axis=1)
y_mat=[]
#raw_y=np.array([1,2,3,4,5])
for k in range(1,11):
	y_mat.append((raw_y==k).astype(int))
#print(y_mat)
#y_mat=[y_mat[-1]]+y_mat[:-1]
y=np.array(y_mat)
#print(y.shape)
'''theta=np.array([2,2,2,2,2])x=np.array([[1,1,1,1,1],[2,2,2,2,2]])print(sigmoid(theta,x).shape)print(theta.shape)print(np.dot(sigmoid(theta,x),x))print(np.dot(x.T,sigmoid(theta,x)))'''
theta=logistic_regression(X,y[0])
#print(theta)
cnt_y=predict(theta,X)
print(np.mean(cnt_y==y[0]))
k_theta=np.array([logistic_regression(X,y[i]) for i in range(10)])
#print(k_theta)
y_maps=np.argmax(sigmoid(k_theta.T, X),axis=1)
print(y_maps)
predict_k(k_theta.T,X)