import os
import numpy as np
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
class WordToVec():
    def __init__(self,k,num_features):
        self.k = k
        self.num_features = num_features
    def DNA2Sentence(self,dna):
        sentence = ""
        length = len(dna)

        for i in range(length - self.k + 1):
            sentence += dna[i: i + self.k] + " "

        sentence = sentence[0: len(sentence) - 1]
        return sentence

    def Get_Unsupervised(self,fname, gname):
        f = np.load(fname, allow_pickle=True)
        g = open(gname, 'w')
        K = self.k
        for i in f:
            if '>' not in i:
                i = i.strip('\n').upper()
                line = self.DNA2Sentence(i)
                g.write(line + '\n')
        g.close()

    def getWord_model(self,word, min_count=1):
        word_model = ""
        if not os.path.isfile("model_3"):
            sentence = LineSentence("2Un", max_sentence_length=29)
            print("Start Training Word2Vec model...")
            # Set values for various parameters
            num_features = int(self.num_features)  # Word vector dimensionality
            min_word_count = int(min_count)  # Minimum word count
            num_workers = 20  # Number of threads to run in parallel并行运行的线程数
            context = 20  # Context window size上下文窗口大小
            downsampling = 1e-3  # Downsample setting for frequent words常用词的下采样设置

            print("Training Word2Vec model...")
            word_model = Word2Vec(sentence, workers=num_workers,
                                  size=num_features, min_count=min_word_count,
                                  window=context, sample=downsampling, seed=1, iter=50, sg=1)
            word_model.init_sims(replace=False)
            word_model.save("model_3")

        else:
            print("Loading Word2Vec model...")
            word_model = Word2Vec.load("model_3")

            return word_model

    def getDNA_split(self, DNAdata):
        DNAlist1 = []
        counter = 0
        for DNA in DNAdata:
            DNA = str(DNA).upper()
            DNAlist1.append(
                self.DNA2Sentence(DNA).split(" "))

            counter += 1
        return DNAlist1

    def getAvgFeatureVecs(self,DNAdata1, model):
        counter = 0
        DNAFeatureVecs = np.zeros((len(DNAdata1), self.num_features), dtype="float32")
        for DNA in DNAdata1:
            if counter % 1000 == 0:
                print("DNA %d of %d\r" % (counter, len(DNAdata1)))
                sys.stdout.flush()
            DNAFeatureVecs[counter][0:self.num_features] = np.mean(model[DNA], axis=0)
            counter += 1
        print()
        return DNAFeatureVecs
if __name__ ==  "__main__":
    word2vec = WordToVec(1, 500)
    word2vec.Get_Unsupervised('test_pos_data.npy', 'pos2Un')
    word2vec.Get_Unsupervised('test_neg_data.npy', 'neg2Un')
    with open('2Un', 'ab') as f:
        f.write(open('pos2Un', 'rb').read())
        f.write(open('neg2Un', 'rb').read())
    data = pd.read_excel("Ub_protein_Test.xlsx")['seq'].values
    datawords1 = word2vec.getDNA_split(data)
    word2vec.getWord_model(2, 1)
    word_model = Word2Vec.load("model_3")
    dataDataVecs = word2vec.getAvgFeatureVecs(datawords1, word_model)
    print(dataDataVecs.shape)
    np.save("WTV1_500_test.npy", dataDataVecs)
