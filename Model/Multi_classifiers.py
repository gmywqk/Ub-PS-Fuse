from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,recall_score,matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
kf = KFold(n_splits=10,shuffle=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, SimpleRNN, Input, GRU
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout, Activation, MaxPooling1D,Conv1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_uniform
def precision(confusion_matrix):
    TP = confusion_matrix[1][1]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[0][0]
    Precision = TP / (TP + FP)
    return Precision
def specific(confusion_matrix):
    TP = confusion_matrix[1][1]
    FN = confusion_matrix[1][0]
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[0][0]
    sp = TN / (TN + FP)
    return sp
def protein_train_features():
    F1 = np.load("../Protein/GO-KNN/KNN_Train.npy")
    F2 = np.load("../Feature_extract/Word2/files/CBOW_features_train_protein_2500.npy")
    F3 = np.load("../Protein/WTV/Skip-Gram_protein_train_2500.npy")
    X = np.concatenate((F1, F2, F3), axis=1)
    print(X.shape)
    return X
def protein_test_features():
    F1 = np.load("../Feature/protein_test/KNN_Test.npy")
    print(F1.shape)
    F2 = np.load("../Feature_extract/Word2/files/CBOW_features_test_protein_2500.npy")
    F3 = np.load("../Protein/WTV/Skip-Gram_protein_test_2500.npy")
    X = np.concatenate((F1, F2, F3), axis=1)
    print(X.shape)
    return X
def train_features():
    features1 = np.load("../Feature/Skip-Gram_site_train_2500.npy")
    features4 = np.load("../Feature/DPC_train.npy")
    features10 = np.load("../Feature/EAAC_train.npy")
    X = np.concatenate((features1, features10), axis=1)
    return X
def test_features():
    feature1test = np.load("../Feature/Test/Skip-Gram_site_test_site_2500.npy")
    feature5test = np.load("../Feature/Test/CBOW_features_test_site.npy")
    print(feature1test.shape)
    feature2test = np.load("../Feature/Test/CKSAAP.npy")
    print(feature2test.shape)
    feature3test = np.load("../Feature/Test/DPC.npy")
    print(feature3test.shape)
    feature4test = np.load("../Feature/Test/EAAC.npy")
    print(feature4test.shape)
    X = np.concatenate((feature1test, feature4test), axis=1)
    # X = feature1test
    return X
def create_dl_model(units):
    '''
    create the deep-learning model
    :return: dl model
    '''
    model = Sequential()
    model.add(Dense(units=500, input_dim=units, kernel_initializer=glorot_uniform))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(units=250, kernel_initializer=glorot_uniform))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(units=125, kernel_initializer=glorot_uniform))
    model.add(LeakyReLU())
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer=glorot_uniform))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def BiLSTM(dim):
    TIME_STEPS = 31
    INPUT_SIZE = dim
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
                                 return_sequences=True, recurrent_initializer='glorot_uniform'), merge_mode='concat'))
    # model.add(Bidirectional(GRU(units=128, batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    #                              return_sequences=True, recurrent_initializer='glorot_uniform'), merge_mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(2e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model
def ML_model():
    count = 0
    accs = []
    recalls = []
    sps = []
    pre = []
    F1_scores = []
    aucs = []
    mccs = []
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        # mode2 = RandomForestClassifier(n_estimators=500)
        mode2 = SVC(probability=True)
        # mode2 = LGBMClassifier(learning_rate=0.1, n_estimators=800, max_depth=4)
        # mode2 = KNeighborsClassifier(n_neighbors=10)

        mode2.fit(X_train, y_train)

        data5_predicted = mode2.predict(X_test)
        probs = mode2.predict_proba(X_test)
        ACC = accuracy_score(y_test, data5_predicted)
        recall = recall_score(y_test, data5_predicted)
        cm = confusion_matrix(y_test, data5_predicted)
        sp = specific(cm)
        mcc = matthews_corrcoef(y_test, data5_predicted)
        Precision = precision(cm)
        fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
        F1_score = (2*Precision*recall)/(Precision+recall)
        Auc = auc(fpr, tpr)
        accs.append(ACC)
        recalls.append(recall)
        sps.append(sp)
        mccs.append(mcc)
        pre.append(Precision)
        F1_scores.append(F1_score)
        aucs.append(Auc)
        print("第{}次,ACC:{}".format(count, ACC))
        print("第{}次,recall:{}".format(count, recall))
        print("第{}次,sp:{}".format(count, sp))
        print("第{}次,mcc:{}".format(count, mcc))
        print("第{}次,pre:{}".format(count, Precision))
        print("第{}次,F1:{}".format(count, F1_score))
        print("第{}次,AUC:{}".format(count, Auc))
        print("-----------------------")
        count += 1
    print("Acc :{}".format(np.mean(accs)))
    print("recall :{}" .format(np.mean(recalls)))
    print("sp :{}" .format(np.mean(sps)))
    print("mcc :{}" .format(np.mean(mccs)))
    print("pre :{}" .format(np.mean(pre)))
    print("F1 :{}" .format(np.mean(F1_scores)))
    print("AUC :{}".format(np.mean(aucs)))
def DL_model(X, y):
    count = 0
    accs = []
    recalls = []
    sps = []
    pre = []
    F1_scores = []
    aucs = []
    mccs = []
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        # mode2 = create_dl_model(X.shape[1])
        mode2 = BiLSTM(X.shape[2])
        mode2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=7)], verbose=1, shuffle=True)
        data5_predicted = mode2.predict(X_test) >= 0.5
        probs = mode2.predict_proba(X_test)
        ACC = accuracy_score(y_test, data5_predicted)
        recall = recall_score(y_test, data5_predicted)
        cm = confusion_matrix(y_test, data5_predicted)
        sp = specific(cm)
        mcc = matthews_corrcoef(y_test, data5_predicted)
        Precision = precision(cm)
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        F1_score = (2 * Precision * recall) / (Precision + recall)
        Auc = auc(fpr, tpr)
        accs.append(ACC)
        recalls.append(recall)
        sps.append(sp)
        mccs.append(mcc)
        pre.append(Precision)
        F1_scores.append(F1_score)
        aucs.append(Auc)
        print("第{}次,ACC:{}".format(count, ACC))
        print("第{}次,recall:{}".format(count, recall))
        print("第{}次,sp:{}".format(count, sp))
        print("第{}次,mcc:{}".format(count, mcc))
        print("第{}次,pre:{}".format(count, Precision))
        print("第{}次,F1:{}".format(count, F1_score))
        print("第{}次,AUC:{}".format(count, Auc))
        print("-----------------------")
        count += 1
    print("Acc :{}".format(np.mean(accs)))
    print("recall :{}".format(np.mean(recalls)))
    print("sp :{}".format(np.mean(sps)))
    print("mcc :{}".format(np.mean(mccs)))
    print("pre :{}".format(np.mean(pre)))
    print("F1 :{}".format(np.mean(F1_scores)))
    print("AUC :{}".format(np.mean(aucs)))
if __name__ == '__main__':
    # data = pd.read_excel("../Protein/Data/Ub_protein_Test.xlsx")
    data = pd.read_excel("../Protein/Data/Ub_protein_Train.xlsx")
    # data = pd.read_excel("../Data/Train.xlsx")
    # data = pd.read_excel("../Data/Test.xlsx")
    y = data['y']
    X = protein_train_features()
    seed = 113
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    # ML_model(X, y)
    X = X.reshape((-1, 1, X.shape[1]))
    DL_model(X, y)
