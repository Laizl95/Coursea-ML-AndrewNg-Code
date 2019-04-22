import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def load_data(path):
	data=scio.loadmat(path)
	#print(data)
	return data['X'],data['y'],data['Xtest'],data['ytest'],data['Xval'],data['yval']

def cal_h(theta,x):
	return np.dot(x,theta.T) #(m，2)*（2，1）(12,1)

def regression_gradient(y,h,X,regular,theta):
	m=X.shape[0]
	deleta=1./m*np.dot((h-y).T,X) #(1,2)
	deleta[:,1:]+=(regular/m)*theta[:,1:]
	return deleta

def cost(y,X,regular,theta):
	m=X.shape[0]
	#print(m)
	h=cal_h(theta,X)
	#print('sss',np.max(regression_gradient(y,h,X,regular,theta)))
	J=np.sum(np.multiply(h-y,h-y))
	#print(J)
	J+=regular*np.sum((np.power(theta,2))[:,1:])
	return (1/(2*m))*J

def linear_regression(y,X,a,regular):
	theta=np.mat(np.ones(X.shape[1]))
	a,t=a,0
	while t<=1000:
		t+=1
		h=np.mat(cal_h(theta,X))
		theta-=a*regression_gradient(y,h,X,regular,theta)
		#print(regression_gradient(y,h,X,regular,theta))
		J=cost(y,X,regular,theta)
		#print(J)
	#print(cost(y,X,regular,theta))
	return theta

def cross_validation(X,y,theta,Xval,yval,a,regular):
	m=X.shape[0]
	traing_cost,cv_cost=[],[]
	for i in range(1,m+1):
		theta=linear_regression(y[:i],X[:i],a,regular)
		J1=cost(y[:i],X[:i],regular,theta)
		J2=cost(yval,Xval,regular,theta)
		traing_cost.append(J1)
		cv_cost.append(J2)
	return traing_cost,cv_cost

def get_ploy_data(X,k):
	poly_data=np.array([np.power(X,i) for i in range(2,k+1)]).reshape(-1,X.shape[0]).T
	#print(X.shape,poly_data.shape)
	return np.concatenate((X,poly_data),axis=1)

def normalize(X):
	a=(X-np.mean(X,axis=0))/np.std(X,axis=0)
	#print('mean',np.mean(X,axis=0))
	return a

def plot_ploly(random_X,theta):#(100,9) (1,9)
	random=get_ploy_data(random_X,6)
	random=np.insert(random,0,np.ones_like(random.shape[0]),axis=1)
	return np.dot(random,theta.T)

path='C:/Users/lai/Desktop/Coursea-ML/code/ex5-bias vs variance/ex5data1.mat'
X,y,Xtest,ytest,Xval,yval=load_data(path)
X=np.insert(X,0,np.ones_like(X.shape[0]),axis=1)
Xval=np.insert(Xval,0,np.ones_like(Xval.shape[0]),axis=1)
Xtest=np.insert(Xtest,0,np.ones_like(Xtest.shape[0]),axis=1)

'''plt.subplot(2,2,1)
plt.scatter(X[:,1],y)
theta=linear_regression(y,X,0.001,0)
x1=np.linspace(-50,50,100)
plt.plot(x1,x1*theta[0,1]+theta[0,0])


plt.subplot(2,2,2)
training_cost,cv_cost=cross_validation(X,y,theta,Xval,yval,0.001,0)
plt.plot(np.arange(1,X.shape[0]+1),np.array(training_cost),label='training cost')
plt.plot(np.arange(1,X.shape[0]+1),cv_cost,label='cv cost')
plt.legend(loc=1)'''

plt.subplot(3,2,1)
X_ploy=normalize(get_ploy_data(X[:,1:],6)) #(12,9)
normal_y=normalize(y)
X_ploy=np.insert(X_ploy,0,np.ones_like(X_ploy.shape[0]),axis=1)
#print(X_ploy.shape,X_ploy)
theta=linear_regression(normal_y,X_ploy,0.1,0)
#print(theta.shape)
random_X=np.linspace(-1.9,2,100).reshape(100,-1)
l=plot_ploly(random_X,theta)
plt.scatter(X_ploy[:,1],normal_y)
plt.plot(random_X,l)

plt.subplot(3,2,2)
Xval_poly=normalize(get_ploy_data(Xval[:,1:],6))
normal_yval=normalize(yval)
Xval_poly=np.insert(Xval_poly,0,np.ones_like(Xval_poly.shape[0]),axis=1)
training_cost,cv_cost=cross_validation(X_ploy,normal_y,theta,Xval_poly,normal_yval,0.01,0)
print(np.array(cv_cost).shape)
plt.plot(np.arange(1,X_ploy.shape[0]+1),np.array(training_cost),label='training cost')
plt.plot(np.arange(1,X_ploy.shape[0]+1),cv_cost,label='cv cost')
plt.legend(loc=1)

plt.subplot(3,2,3)
theta=linear_regression(normal_y,X_ploy,0.1,1)
random_X=np.linspace(-1.9,2,100).reshape(100,-1)
l=plot_ploly(random_X,theta)
plt.scatter(X_ploy[:,1],normal_y)
plt.plot(random_X,l)

plt.subplot(3,2,4)
training_cost,cv_cost=cross_validation(X_ploy,normal_y,theta,Xval_poly,normal_yval,0.01,1)
print(np.array(cv_cost).shape)
plt.plot(np.arange(1,X_ploy.shape[0]+1),np.array(training_cost),label='training cost')
plt.plot(np.arange(1,X_ploy.shape[0]+1),cv_cost,label='cv cost')
plt.legend(loc=1)

plt.subplot(3,2,5)
theta=linear_regression(normal_y,X_ploy,0.01,20)
random_X=np.linspace(-1.9,2,100).reshape(100,-1)
l=plot_ploly(random_X,theta)
plt.scatter(X_ploy[:,1],normal_y)
plt.plot(random_X,l)

plt.subplot(3,2,6)
training_cost,cv_cost=cross_validation(X_ploy,normal_y,theta,Xval_poly,normal_yval,0.01,20)
print(np.array(cv_cost).shape)
plt.plot(np.arange(1,X_ploy.shape[0]+1),np.array(training_cost),label='training cost')
plt.plot(np.arange(1,X_ploy.shape[0]+1),cv_cost,label='cv cost')
plt.legend(loc=1)
plt.show()
