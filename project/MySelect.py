import numpy as np
from sklearn.feature_selection import SelectKBest,chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor

def SelectByVif(train_X,test_X,feature_name):
    feature_vif = []
    X = np.c_[train_X, np.ones(train_X.shape[0])]
    i=0
    while(i<train_X.shape[1]):
        t=variance_inflation_factor(X, i)
        if variance_inflation_factor(X, i) >= 10000:
            train_X=np.delete(train_X,i,axis=1)
            test_X = np.delete(test_X, i, axis=1)
            feature_name = np.delete(feature_name, i)
            X=np.delete(X,i,axis=1)
        else:
            i+=1
    return  train_X,test_X,feature_name

def SelectByChi2(train_X,train_label,test_X,feature_name,k):
    model1 = SelectKBest(score_func=chi2, k=k)
    model1.fit(train_X, train_label)
    train_X = model1.transform(train_X)
    test_X = test_X[:, model1.get_support(indices=True)]
    feature_name =feature_name[model1.get_support(indices=True)]
    return train_X,test_X,feature_name