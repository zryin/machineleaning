from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer
import numpy as np
from NB import trainNB
from NB import predictNB

Vectorizer=CountVectorizer(encoding='latin-1')
mails_train = load_files('E:/PY/email/train')
mails_test = load_files('E:/PY/email/test')

X_train=Vectorizer.fit_transform(d for d in mails_train.data)
Y_train=mails_train.target
arrayP,Y_P=trainNB(X_train,Y_train)


X_test=Vectorizer.transform(d for d in mails_test.data)
Y_test=mails_test.target
print(X_test.shape[0])
RightCount=0
for d in range(X_test.shape[0]):
    if(Y_test[d]==predictNB(X_test[d].toarray(),arrayP,Y_P)):
        RightCount+=1
print('准确率=%.2f%%' % (float(RightCount)/X_test.shape[0]*100))


