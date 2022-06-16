#Extract_GO
import requests
import os
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore")
from requests.packages import urllib3
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import SearchIO
from Bio.Blast import NCBIXML

yb = pd.read_excel("../Qiu_Data/Train_Neg2.xlsx")
bh = list(yb["TrainID_Neg2"])[1500:]
xulie = list(yb["seq"])[1500:]
# bh = ["P0DMP9","Q16661","Q9UBF2","Q91N56"]
datall = {}
for i in range(len(bh)):
    url = "https://www.uniprot.org/uniprot/{}.txt".format(bh[i])
    respones = requests.get(url=url)
    page_text = respones.text
    go = set()
    with open("{}.txt".format(i),"w") as fp:
        fp.write(page_text)
    filename = open("{}.txt".format(i),"r")
    for lines in filename:
        tz = re.findall(r'GO; GO:.*; ',lines)
        if tz == [ ]:
            continue
        tz = tz[0]
        a = "".join(tz)
        a = a[7:14]
        go.add(a)

    if go == set():   # 寻找同源蛋白的go
        id = bh[i]
        seq = xulie[i]
        K = np.array(id).reshape((1, 1))
        V = np.array(seq).reshape((1, 1))
        data1 = np.hstack((K, V))
        data = list(np.arange(1))
        data1 = pd.DataFrame(data1, index=data, columns=['meta', 'seq'])
        data1.to_excel("{}.xlsx".format(bh[i]))
        path = "{}.xlsx".format(bh[i])
        data = pd.read_excel(path)
        file_path = path.replace('.xlsx', '.fasta')
        with open(file_path, "a") as fr:
            id = bh[i]
            seq = xulie[i]
            fr.write(">{}".format(id))
            fr.write("\n")
            fr.write(seq + "\n")
        os.remove(r"{}.xlsx".format(bh[i]))

        blastp_cline = NcbiblastpCommandline(query="{}.fasta".format(bh[i]), db="D:/blast/blast-2.11.0+/db/swissprot", outfmt="5", out="out.xml")
        stdout, stderr = blastp_cline()
        seq = []
        result_handle = open("out.xml")
        blast_records = NCBIXML.parse(result_handle)

        record_list = list(blast_records)
        seq_name = []
        for record in record_list:

            for j in record.alignments:
                new_i_1 = j.hit_id.split("|")[1]
                new_i_2 = new_i_1.split(".")[0]
                seq_name.append(new_i_2)
            seq.append(seq_name)
        # print(seq_name)
        for k in range(len(seq_name)):
            go1 = set()
            url = "https://www.uniprot.org/uniprot/{}.txt".format(seq_name[k])
            # print("{}.txt".format(seq_name[k]))
            respones = requests.get(url=url)
            page_text = respones.text
            go1 = set()
            with open("{}.txt".format(seq_name[k]), "w") as fp:
                fp.write(page_text)
            filename = open("{}.txt".format(seq_name[k]), "r")
            for lines in filename:
                tz = re.findall(r'GO; GO:.*; ', lines)
                if tz == []:
                    continue
                tz = tz[0]
                a1 = "".join(tz)
                a1 = a1[7:14]
                go1.add(a1)
            if go1 != set():
                break
        go = go1
        # if os.path.exists(r"{}.txt".format(seq_name[k])):
        #     os.remove(r"{}txt".format(seq_name[k]))

    datall[bh[i]] = go
    filename.close()
    if os.path.exists(r"{}.txt".format(i)):
        os.remove(r"{}.txt".format(i))

L = len(datall.keys())
K = np.array(list(datall.keys())).reshape((L, 1))
V = np.array(list(datall.values())).reshape((L, 1))
data1 = np.hstack((K, V))
data = list(np.arange(L))
data1 = pd.DataFrame(data1, index=data, columns=['BH', 'GO'])
data1.to_excel("../Qiu_Data/Train_Neg2_GO/Train_Neg2_GO-1500-.xlsx")
