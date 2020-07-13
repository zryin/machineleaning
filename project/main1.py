import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,recall_score
from BAYES import GAUSSIAN,N_GAUSSIAN
import matplotlib.pyplot as plt


n=2000

p=0.7

R=np.zeros([4,6])
arr=np.arange(500,3500,500)
print(arr)

for i in range(100):
    for n in arr:
        read_data = pd.read_csv('E:/PY/voice/voice.csv')
        data = read_data.sample(n)
        train_data = data.sample(frac=p)
        test_data = data.drop([d for d in train_data.index])
        train_label = train_data['label'].values
        train_X = train_data.drop(['label'], axis=1).values
        feature_name=data.drop(['label'],axis=1).columns.values

        test_label = test_data['label'].values
        test_X = test_data.drop(['label'], axis=1).values

        male_train=train_X[train_label=='male']#统计量计算
        male_count=male_train.shape[1]
        male_mean=male_train.mean(axis=0)
        male_var=male_train.var(axis=0)
        male_cov=np.cov(male_train,rowvar=False)

        female_train=train_X[train_label=='female']
        female_count=female_train.shape[1]
        female_mean=female_train.mean(axis=0)
        female_var=female_train.var(axis=0)
        female_cov=np.cov(female_train,rowvar=False)

        male_prior=male_count/train_X.shape[1]
        female_prior=female_count/train_X.shape[1]

        predict=GAUSSIAN(male_mean,male_var,male_prior,female_mean,female_var,female_prior,test_X)

        test_y=np.zeros(len(test_label))
        test_y[(test_label=='male')]=0
        test_y[(test_label=='female')]=1
        y_label=['male','female']
        prec=precision_score(test_y,predict,average=None)
        rec=recall_score(test_y,predict,average=None)
        R[0,np.where(arr==n)]+=prec[0]
        R[2,np.where(arr == n)]+= prec[1]
        R[1,np.where(arr == n)]+= rec[0]
        R[3,np.where(arr == n)]+= rec[1]
R=R/100
ax1 = plt.subplot(1,1,1,facecolor='white')
plt.rcParams['font.sans-serif']=['SimHei']
plot1=ax1.plot(arr,R[0],linestyle='-',alpha=0.5,color='r',label='男声精确率')
plot2=ax1.plot(arr,R[1],linestyle='-',alpha=0.5,color='y',label='男声召回率')
plot3=ax1.plot(arr,R[2],linestyle='-',alpha=0.5,color='b',label='女声精确率')
plot4=ax1.plot(arr,R[3],linestyle='-',alpha=0.5,color='g',label='女声召回率')

ax1.set_title('男声女声预测评估与数据量关系统计图')
ax1.set_xlabel('数据总数')

ax1.set_xlim(500,3000)
ax1.set_ylim(0.8,1)
plt.legend()
plt.show()

