from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
# nghich dao vec to X
#X = [1,2,3] mang 1 chieu
# Aco 2 hang 3 cot 
#print(X.shape)
#A = np.array([[1, 2], [3, 4]])Hang,cot
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T #hang co 13phan tu, cot k co
#nghich dao thanh (13 hang  co 1 cot co ) 
#print(X.shape)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
# Building Xbar 
#print(X.shape[0]) # so chieu la 13
one = np.ones((X.shape[0], 1)) #khoi tao mang one la 1 het: 13 so 1

Xbar = np.concatenate((one, X), axis = 1) # noi hang 
print(Xbar)
# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar) # tinh A  bang X nganh nghich dao * X ngang// "dot" la dau cham la phep tinh nhan
print(A)
b = np.dot(Xbar.T, y) 
w = np.dot(np.linalg.pinv(A), b) # tinh ra w, ham pinv tinh ma tran gia nghich dao cua A
print(w.shape)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]# w0 la mac dinh, la sai so du doan
w_1 = w[1][0]# do du lieu 1 chieu (chieu cao de du dao can nang) nen co w1,
x0 = np.linspace(145, 185, 2) # buoc nhay so nay den so kia la 2
y0 = w_0 + w_1*x0       # ket qua du doan 
print(y0)
# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
#plt.show()

from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
print(Xbar.shape)
X = np.array([[1.0, 160]])
print('Predict Height:', regr.predict(X))
# viec hoc se xac dinh cac w
# khi co nhieu data thi fit
#y1 = w_1*155 + w_0
#y2 = w_1*160 + w_0
#X=(155,160)#datainput du doan 2 nguoi
#Y=(1,y2) ket qua du doan cua 2 nguoi
# du doan nhieu ng thi tao matran X sau do tinh y[i]=w_1*X[i]+w_0
#print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )