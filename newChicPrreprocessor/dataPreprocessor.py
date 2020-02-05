import os
import sys
# import tensorflow as tf
import json

file_path = "/home/jugs/PycharmProjects/ExperimentalProjects/new_chic/DataFeedNewchicDatafeed.json" # "/home/jugs/Downloads/DataFeedNewchicDatafeed.json"

with open(file_path, "r") as fp:
    data = json.load(fp)

categories = set()
catDict = {}
newChicData = []
for i, rec in enumerate(data):
    recsplit = rec['Category'].split('>')
    if recsplit[1].strip() not in catDict:
        catDict[recsplit[1].strip()] = len(catDict)
    var = rec['Name'].replace('\n', '').strip() + '. ' + rec['Description'].replace('\n', '').strip()
    categories.add(recsplit[1].strip())
    line = str(i) + '\t' + recsplit[1].strip() + '\t' + str(catDict[recsplit[1].strip()]) + '\t' + var
    newChicData.append(line)

with open("newChicData.tsv", 'w') as fp:
    for line in newChicData:
        fp.write(line + '\n')
    fp.close()

with open("labels.txt", "w") as fp:
    for line in categories:
        fp.write(line.strip() + '\n')
    fp.close()

with open("catDict.txt", "w") as fp:
    for k,v in catDict.items():
        fp.write(k + '\t' + str(v) + '\n')
    fp.close()

print(len(categories))
print(len(data))
print("Tested OK")

