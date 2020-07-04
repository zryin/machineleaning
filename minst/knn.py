from numpy import *
import operator


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   # shape[0]表示行数


    diff = tile(newInput, (numSamples, 1)) - dataSet
    addDiff = abs(diff)
    distance = sum(addDiff, axis = 1)
    sortedDistIndices = argsort(distance)
    classCount = {} # define a dictionary (can be append element)
    for i in range(0,k):
        # # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # # step 4: 计算k个最近邻中各类别出现的次数
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex