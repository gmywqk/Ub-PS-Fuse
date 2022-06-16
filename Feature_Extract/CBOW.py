import warnings
warnings.filterwarnings('ignore')
from gensim.models import word2vec, Word2Vec
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
import os
import pandas as pd
from tqdm import tqdm

def get_text(path):
    for i in range(1, 6):
        path = str(path)
        data = pd.read_excel(path)
        file_name = "all_{}.txt".format(i)
        file_path = path.replace("Test.xlsx", file_name)
        print(file_path)
        with open(file_path, "w") as fr:
            for j in tqdm(data.seq):
                text = ""
                for k in range(len(j) - i + 1):
                    if k != len(j) - i:
                        text = text + j[k: k+i] + " "
                    else:
                        text = text + j[k: k+i]
                fr.write(text + "\n")

def train_model():
    '''
    train the CBOW models
    '''
    print("train cbow models...")
    get_text(r'files/Test.xlsx')
    print("train cbow models 1 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'files/all_1.txt')
    word2vec.Word2Vec(sentences=sentences, min_count=0, size=300, window=5, compute_loss=True, iter=10, workers=8, sg=0, callbacks=[EpochLogger("l1")])
    print("train cbow models 2 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'files/all_2.txt')
    word2vec.Word2Vec(sentences=sentences, min_count=0, size=300, window=5, compute_loss=True, iter=10, workers=8, sg=0, callbacks=[EpochLogger("l2")])
    print("train cbow models 3 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'files/all_3.txt')
    word2vec.Word2Vec(sentences=sentences, min_count=0, size=300, window=5, compute_loss=True, iter=10, workers=8, sg=0, callbacks=[EpochLogger("l3")])
    print("train cbow models 4 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'files/all_4.txt')
    word2vec.Word2Vec(sentences=sentences, min_count=0, size=300, window=5, compute_loss=True, iter=10, workers=8, sg=0, callbacks=[EpochLogger("l4")])
    print("train cbow models 5 .. Please wait patiently")
    sentences = word2vec.Text8Corpus(r'files/all_5.txt')
    word2vec.Word2Vec(sentences=sentences, min_count=0, size=300, window=5, compute_loss=True, iter=10, workers=8, sg=0, callbacks=[EpochLogger("l5")])
  

def get_features(file_path, model):
    '''
    extract the CBOW features
    '''
    f = open(file_path, 'r')
    content = f.readlines()
    f.close()
    count = 0
    features = None
    for line in tqdm(content):
        line = line.split()
        single_feature = np.mean(model[line], axis=0, keepdims=True)
        if count == 0:
            features = single_feature
            count += 1
        else:
            features = np.r_[features, single_feature]

    return features


def merge_features():
    '''
    concatenate the three kinds CBOW features
    '''
    train_model()
    print("extract cbow features...")
    if os.path.exists(r'files/l1.model.ckpt'):
        model_1 = Word2Vec.load(r'files/l1.model.ckpt')
    else:
        model_1 = Word2Vec.load(r'files/last_l1.model.ckpt')
    if os.path.exists(r'files/l2.model.ckpt'):
        model_2 = Word2Vec.load(r'files/l2.model.ckpt')
    else:
        model_2 = Word2Vec.load(r'files/last_l2.model.ckpt')
    if os.path.exists(r'files/l3.model.ckpt'):
        model_3 = Word2Vec.load(r'files/l3.model.ckpt')
    else:
        model_3 = Word2Vec.load(r'files/last_l3.model.ckpt')
    if os.path.exists(r'files/l4.model.ckpt'):
        model_4 = Word2Vec.load(r'files/l4.model.ckpt')
    else:
        model_4 = Word2Vec.load(r'files/last_l4.model.ckpt')
    if os.path.exists(r'files/l5.model.ckpt'):
        model_5 = Word2Vec.load(r'files/l5.model.ckpt')
    else:
        model_5 = Word2Vec.load(r'files/last_l5.model.ckpt')
    # if os.path.exists(r'files/l6.model.ckpt'):
    #     model_6 = Word2Vec.load(r'files/l6.model.ckpt')
    # else:
    #     model_6 = Word2Vec.load(r'files/last_l6.model.ckpt')
    # if os.path.exists(r'files/l7.model.ckpt'):
    #     model_7 = Word2Vec.load(r'files//l7.model.ckpt')
    # else:
    #     model_7 = Word2Vec.load(r'files/last_l7.model.ckpt')
    feautures_1 = get_features(r'files/all_1.txt', model_1.wv)
    feautures_2 = get_features(r'files/all_2.txt', model_2.wv)
    feautures_3 = get_features(r'files/all_3.txt', model_3.wv)
    feautures_4 = get_features(r'files/all_4.txt', model_4.wv)
    feautures_5 = get_features(r'files/all_5.txt', model_5.wv)
    # feautures_6 = get_features(r'files/all_6.txt', model_6.wv)
    # feautures_7 = get_features(r'files/all_7.txt', model_7.wv)
    # CBOW_features = feautures_7
    CBOW_features = np.concatenate([feautures_1, feautures_2, feautures_3, feautures_4, feautures_5], axis=1)
    np.save(r'files/CBOW_features_protein_test_Qiu.npy', CBOW_features)


class EpochLogger(CallbackAny2Vec):
    def __init__(self, name):
        self.epoch = 1
        self.losses = []
        self.previous_losses = 0
        self.add_losses = []
        self.name = name

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        self.epoch += 1
        self.add_losses.append(loss - self.previous_losses)
        self.previous_losses = loss
        if self.epoch > 2:
            if self.add_losses[-1] == 0 and self.add_losses[-2] != 0:
                path = r'files/' + os.sep + "{}.model.ckpt".format(self.name)
                model.save(path)

    def on_train_end(self, model):
        if self.add_losses[-1] == 0:
            pass
        else:
            path = r"files/" + os.sep + "last_{}.model.ckpt".format(self.name)
            print(path)
            model.save(path)

if __name__ == '__main__':
    path = r'files/Test.xlsx'
    # get_text(path)
    merge_features()
