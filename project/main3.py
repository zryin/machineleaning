import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from BAYES import GAUSSIAN,N_GAUSSIAN
import matplotlib.pyplot as plt
from MySelect import SelectByChi2,SelectByVif


n=3000

p=0.7

R=np.zeros(17)
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
    print(train_X.shape)
    for k in range(1,train_X.shape[1]+1):
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

        predict = GAUSSIAN(male_mean, male_var, male_prior, female_mean, female_var, female_prior, select_test_X)
        test_y = np.zeros(len(test_label))
        test_y[(test_label == 'male')] += 0
        test_y[(test_label == 'female')] += 1
        y_label = ['male', 'female']
        R[k - 1] += precision_score(test_y, predict, average='weighted')
    print(i)
R=R/100
print(R)
ax1 = plt.subplot(1,1,1,facecolor='white')
plt.rcParams['font.sans-serif']=['SimHei']
x_index = np.arange(17)+1
bar_width=0.35
rects1 = plt.bar(x_index, R, width=bar_width,alpha=0.4, color='r')
plt.xticks(x_index, np.arange(1,18,1))
plt.tight_layout()
ax1.set_title('声音预测准确率与特征选择特征数关系')#'多维高斯分布和高斯朴素贝叶斯对比'
ax1.set_xlabel('特征选择特征数')
ax1.set_ylabel('准确率')

ax1.set_ylim(0.5,1)
plt.show()