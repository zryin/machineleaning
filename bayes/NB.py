import numpy as np

def trainNB(X_train,Y_train):
    arrayP=np.zeros([np.max(Y_train)+1,X_train.shape[1]])
    Y_P=np.zeros(np.max(Y_train)+1)
    for d in range(len(Y_train)):
        arrayP[Y_train[d]]+=X_train[d]
        Y_P[Y_train[d]]+=1
    arrayP+=1
    arrayP=np.log(arrayP/np.transpose(np.repeat([(arrayP.sum(axis=1))],arrayP.shape[1],axis=0)))
    Y_P=np.true_divide(Y_P,sum(Y_P))
    return arrayP,Y_P

def predictNB(X_test,arrayP,Y_P):
    vec1=(np.multiply(arrayP,X_test))
    vec=vec1.sum(axis=1)
    print(vec)
    vec+=np.log(Y_P)
    return np.argmax(vec)
