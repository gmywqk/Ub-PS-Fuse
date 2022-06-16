import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
def CalculateDistance(sequence1, sequence2):
    a = sequence1.replace("{", "")
    a = a.replace("}", "")
    a = a.replace("'", '')
    a = a.split(", ")
    sequence1 = set(a)
    a = sequence2.replace("{", "")
    a = a.replace("}", "")
    a = a.replace("'", '')
    a = a.split(", ")
    sequence2 = set(a)
    distance = 1 - (len(list(sequence1 & sequence2)) / len(list(sequence1 | sequence2)))
    return distance


# 计算前J个样本中正样本比例
def CalculateContent(myDistance, kNum):
    content = np.zeros(len(kNum))
    # 记录前J个正负样本个数
    for i in range(len(kNum)):
        k = kNum[i]
        num_occurances = defaultdict(int)
        for index in range(k):
            if myDistance[index][0] == 0:
                num_occurances[0] += 1
        num = list(num_occurances.values())
        if num == []:
            num = 0
        else:
            num = num[0]
        content[i] = num / k
    return content


def knn_feature(seq_r, pos_r, seq):
    # positive 0, negative 1
    myLabel = np.zeros(seq_r)
    myLabel[pos_r:] = 1

    kNum = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    encodings = np.zeros((seq_r, 11))

    # knndir = os.path.join(os.path.dirname(__file__), '../dataset/knn/')
    # os.chdir(knndir)
    # 保存所有序列之间的距离，上三角矩阵，每个元素包含[标签，距离]
    myDistance = np.zeros((seq_r, (seq_r-1), 2))
    for i in range(seq_r):
        sequence = seq.iloc[i]
        for j in range((seq_r-1)):
            if i > j:
                myDistance[i, j] = [myLabel[j], CalculateDistance(seq.iloc[j], sequence)]
            else:
                myDistance[i,j] = [myLabel[(j+1)], CalculateDistance(seq.iloc[(j+1)], sequence)]
    # 保存序列之间距离矩阵
    np.save('distance_matrix.npy', myDistance)

    # 获取序列之间距离矩阵
    myDistance = np.load('distance_matrix.npy')


    # distance = np.zeros(((seq_r - 1), 2))
    for i in range(seq_r):
        distance = myDistance[i, :]
        distance = distance[np.argsort(distance[:, 1])]
        encodings[i] = CalculateContent(distance, kNum)

    data_pos = encodings[:pos_r]
    data_neg = encodings[pos_r:]
    return data_pos, data_neg

pos_data = pd.read_excel("Qiu_GO_Data/Train_Pos_GO.xlsx")
neg_data = pd.read_excel("Qiu_GO_Data/Train_Neg3_GO.xlsx")

pos_seq = pos_data.iloc[:, 2]
neg_seq = neg_data.iloc[:, 2]
seq = pd.concat([pos_seq, neg_seq], axis=0, ignore_index=True)

pos_r = pos_seq.shape[0]
neg_r = neg_seq.shape[0]
seq_r = seq.shape[0]

pos_feature, neg_feature = knn_feature(seq_r, pos_r, seq)

# 保存特征
# pos_dir = os.path.join(os.path.dirname(__file__), '../dataset/pos/')
# os.chdir(pos_dir)
np.save('Qiu_pos_knn_feature_train3.npy', pos_feature)
# neg_dir = os.path.join(os.path.dirname(__file__), '../dataset/neg/')
# os.chdir(neg_dir)
np.save('Qiu_neg_knn_feature_train3.npy', neg_feature)

if __name__ == '__main__':
    a = np.load("Qiu_pos_knn_feature_train3.npy")
    b = np.load("Qiu_neg_knn_feature_train3.npy")
    c = np.concatenate((a, b), axis=0)
    np.save("Qiu_KNN_Train3.npy", c)
