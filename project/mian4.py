import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from BAYES import GAUSSIAN,N_GAUSSIAN
import matplotlib.pyplot as plt
from MySelect import SelectByChi2,SelectByVif
from time import time


n=3000

p=0.7
t=np.zeros([2,5])
R=np.zeros(5)
RN=np.zeros(5)
read_data = pd.read_csv('E:/PY/voice/voice.csv')

for i in range(100):
    data = read_data.sample(n)
    train_data = data.sample(frac=p)
    test_data = data.drop([d for d in train_data.index])
    train_label = train_data['label'].values
    train_X = train_data.drop(['label'], axis=1).values
    feature_name=data.drop(['label'],axis=1).columns.values

    test_label = test_data['label'].values
    test_X = test_data.drop(['label'], axis=1).values
    train_X,test_X,feature_name=SelectByVif(train_X, test_X, feature_name)
    for k in range(8,13):
        select_train_X, select_test_X, select_feature = SelectByChi2(train_X, train_label, test_X, feature_name, k)

        male_train = select_train_X[train_label == 'male']  # 统计量计算
        male_count = male_train.shape[1]
        male_mean = male_train.mean(axis=0)
        male_var = male_train.var(axis=0)
        male_cov = np.cov(male_train, rowvar=False)

        female_train = select_train_X[train_label == 'female']
        female_count = female_train.shape[1]
        female_mean = female_train.mean(axis=0)
        female_var = female_train.var(axis=0)
        female_cov = np.cov(female_train, rowvar=False)

        male_prior = male_count / select_train_X.shape[1]
        female_prior = female_count / select_train_X.shape[1]
        start=time()
        predict = GAUSSIAN(male_mean, male_var, male_prior, female_mean, female_var, female_prior, select_test_X)
        stop1=time()
        t[0,k-8]+=stop1-start
        predictN = N_GAUSSIAN(male_mean, male_cov, male_prior, female_mean, female_cov, female_prior, select_test_X)
        stop2=time()
        t[1,k-8]+=stop2-stop1
        test_y = np.zeros(len(test_label))
        test_y[(test_label == 'male')] = 0
        test_y[(test_label == 'female')] = 1
        y_label = ['male', 'female']
        R[k - 8] += precision_score(test_y, predict, average='weighted')
        RN[k - 8] += precision_score(test_y, predictN, average='weighted')
    print(i)
R=R/100
RN=RN/100
t=t/100
print(t)
ax1 = plt.subplot(1,1,1,facecolor='white')
plt.rcParams['font.sans-serif']=['SimHei']
x_index = np.arange(5)+1
bar_width=0.35
rects1 = plt.bar(x_index, R, width=bar_width,alpha=0.4, color='r',label='高斯朴素贝叶斯')
rects1 = plt.bar(x_index+bar_width, RN, width=bar_width,alpha=0.4, color='y',label='多维高斯贝叶斯')
plt.xticks(x_index+bar_width/2, range(8,13))
plt.tight_layout()
ax1.set_title('多维高斯分布和高斯朴素贝叶斯对比')
ax1.set_xlabel('特征选择特征数')
ax1.set_ylabel('准确率')
ax1.set_ylim(0.8,1)
plt.show()