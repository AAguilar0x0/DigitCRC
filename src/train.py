from mls import LogisticRegression, NeuralNetwork, SupportVectorMachine
import pickle
import time
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt



with open('mnistdataset.dat','rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)
X_train = X_train / 255.0
X_test = X_test / 255.0

y = np.atleast_2d(1*(range(10) == y_train))



# =============================================================================================================================

a = 0.0001 # 0.00001
l = 0.01
num_iters = X_train.shape[0] * 1



######################### Neural Network #########################

nn = NeuralNetwork(X_train.shape[1],25,10,X_train,y)
print("Training Neural Network")
start = time.time()
nn_w,nn_b,nn_cost = nn.adam(num_iters,y)
end = time.time()
print(f"Training took: {end-start}(s)")



del(nn._X,nn._y)
with open('models/nn.model','wb') as f:
    pickle.dump(nn,f)



######################### Logistic Regression #########################

lr = LogisticRegression(X_train,y_train)
lr_w_all = np.zeros((X_train.shape[1],10))
lr_b_all = np.zeros((1,10))
fig, axs = plt.subplots(2,5)
print("Training Logistic Regression")
start = time.time()
for i in range(10):
    w,b,cost = lr.adam(num_iters,np.atleast_2d(y[:,i]).T,alpha=a)
    lr_w_all[:,i] = w[:,0]
    lr_b_all[0,i] = b
    axs[0 if i < 5 else 1, i if i < 5 else (i - 5)].plot(range(num_iters),cost,label=str(i))
    axs[0 if i < 5 else 1, i if i < 5 else (i - 5)].legend()
end = time.time()
lr.w = lr_w_all
lr.b = lr_b_all
print(f"Training took: {end-start}(s)")



del(lr._X,lr._y)
with open('models/lr.model','wb') as f:
    pickle.dump(lr,f)



######################### Support Vector Machine #########################

m = 10000 # X_train.shape[0]
svm_max_passes = 10
svm_sigma = None
svm_C = 1
svm_tol = 1e-20
svm = [ SupportVectorMachine(X_train[:m,:], np.atleast_2d(y[:m,i]).T, 'gaussian') for i in range(10) ]
print("Training Support Vector Machine")
k_start = time.time()
svm[0].calculate_K(0.3)
k_end = time.time()
start = time.time()
svm[0].smo(max_passes=svm_max_passes,sigma=svm_sigma,C=svm_C,tol=svm_tol)
for i in range(1,10):
    svm[i].K = svm[0].K
    svm[i].sigma = svm[0].sigma
    svm[i].smo(max_passes=svm_max_passes,sigma=svm_sigma,C=svm_C,tol=svm_tol)
end = time.time()
print(f"Loading K took: {k_end-k_start}(s)")
print(f"Training took: {end-start}(s)")



for i in range(10):
    del(svm[i]._K,svm[i]._E)
with open('models/svm.model','wb') as f:
    pickle.dump(svm,f)



########## End of Training ##########



print("Neural Network Weight and Bias: ",nn_w.shape,nn_b.shape)
print("Logistic Regression Weight and Bias: ",lr_w_all.shape,lr_b_all.shape)



print("\n\nTest:\n")

print("Predicting (Neural Network)")
start = time.time()
p = nn.predict(X_test)
end = time.time()
print(f"Predicting took {end-start}")
print("Confusion Matrix:")
print(confusion_matrix(y_test.ravel(),p[:,1].ravel()))
print("Classification Report:")
print(classification_report(y_test.ravel(),p[:,1].ravel()))

print("Predicting (Logistic Regression)")
start = time.time()
p = lr.predict_onevsall(X_test)
end = time.time()
print(f"Predicting took {end-start}")
print("Confusion Matrix:")
print(confusion_matrix(y_test.ravel(),p[:,1].ravel()))
print("Classification Report:")
print(classification_report(y_test.ravel(),p[:,1].ravel()))

print("Predicting (Support Vector Machine)")
svm_test_m = X_test.shape[0]
start = time.time()
p = np.array([ svm[i].predict(X_test[:svm_test_m,:])[:,0] for i in range(10) ]).T
pred = np.zeros((svm_test_m,2))
pred[:,0] = p.max(axis=1)
pred[:,1] = p.argmax(axis=1)
end = time.time()
print(f"Prediction took {end-start}")
print("Confusion Matrix:")
print(confusion_matrix(y_test[:svm_test_m,0].ravel(),pred[:,1].ravel()))
print("Classification Report:")
print(classification_report(y_test[:svm_test_m,0].ravel(),pred[:,1].ravel()))



plt.figure()
plt.plot(range(1,nn_cost.shape[0]+1),nn_cost[:,0],label="Neural Network Train Cost")
plt.legend()
plt.show()