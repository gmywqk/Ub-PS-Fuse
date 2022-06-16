import re, sys, os
from collections import Counter
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import numpy as np
import pandas as pd

def EAAC(a):
    window = 5
    encodings = []
    code = []
    AA = 'ACDEFGHIKLMNPQRSTVWYX'
    for j in range(len(a)):
        if j < len(a) and j + window <= len(a):
            count = Counter(re.sub('-', '', a[j:j+window]))
            for key in count:
                count[key] = count[key] / len(a[j:j+window])
            for aa in AA:
                code.append(count[aa])
    encodings.append(code)
    return np.array(encodings)

kw = {'path': r"Training_data.txt",'order': 'ACDEFGHIKLMNPQRSTVWY'}
