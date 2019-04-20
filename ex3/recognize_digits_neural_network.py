import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def load_weight(path):
	data=scio.loadmat(path)
	return data['Theta1'],data['Theta2']
def load_data(path):
	data=scio.loadmat(path)
	return data['X'],data['y']
def sigmoid(z):
	return 1./(1+np.exp(-z))
def get_y(y):
	y=np.reshape(y,y.shape[0])
	y_mat=[]
	for k in range(1,11):
		y_mat.append((y==k).astype(int))
	y=np.array(y_mat)
	return y
def predict(y,h):
	h_maps=np.argmax(h,axis=1)
    
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex3/ex3weights.mat'
theta1,theta2=load_weight(path)
path='C:/Users/lai/Desktop/Coursea机器学习作业/代码/ex3/ex3data1.mat'
X,y=load_data(path)
X=np.insert(X,0,values=np.ones_like(X.shape[0]),axis=1)
print(X.shape,' ',y.shape)
print(theta1.shape,' ',theta2.shape)
z2=np.dot(X,theta1.T)
print('z2',z2.shape)
z2=np.insert(z2,0,values=np.ones_like(z2.shape[0]),axis=1)
b2=sigmoid(z2)
z3=np.dot(b2,theta2.T)
a3=sigmoid(z3)
y=get_y(y)
cnt_y=(np.array([a3[:,i] for i in range(a3.shape[1])])>=0.5).astype(int)
print(cnt_y.shape)
p=np.mean(np.array([cnt_y[i,:]==y[i] for i in range(10)]).astype(int),axis=1)
print(p)

