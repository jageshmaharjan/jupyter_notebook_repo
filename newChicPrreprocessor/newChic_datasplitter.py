import os
import sys
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

filepath = "/home/jugs/PycharmProjects/ExperimentalProjects/new_chic/newChicData.tsv"

dataframe = pd.read_csv(filepath, delimiter='\t')
print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.2, shuffle=True)
train, val = train_test_split(train, test_size=0.2, shuffle=True)

print(len(train))
print(len(test))
print(len(val))

for line in train:
    print(line)

def create_records(dataframe):
    datarec = []
    for index, row in dataframe.iterrows():
        datarec.append(str(row['guid']) + '\t' + row['label'] + '\t' + str(row['labelcode']) + '\t' + row['sentenes'])
    return datarec

def write_tsv(datalist, filename):
    with open(filename, "w") as fp:
        for line in datalist:
            fp.write(line + '\n')
        fp.close()

trainrec = create_records(train)
testrec = create_records(test)
valrec = create_records(val)
write_tsv(trainrec, "train.tsv")
write_tsv(testrec, "test.tsv")
write_tsv(valrec, "dev.tsv")
