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
	print(X[4,:].shape)
	print(sample_images[2].shape)
	print(sample_images.shape)
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
print(y_mat)	
print(y_mat[-1:])
y_mat=[y_mat[-1]]+y_mat[:-1]
#print(y_mat[1:])
y=np.array(y_mat)
print(y)

