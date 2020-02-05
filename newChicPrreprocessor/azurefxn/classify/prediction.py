import json
import requests
import numpy as np
from . import tokenization
import argparse
import urllib.request
import os

def get_query(query):
  return query + "now new query"


def predict(query):
    query = query #request.args.get('text')

    endpoints = "http://18.162.113.148:8501/v1/models/newchicclassifier:predict"
    headers = {"content-type": "application-json"}

    path = os.getcwd()
    path = str(path)
    filepath = os.path.join(path,'classify/vocab.txt')

    tokenizer = tokenization.FullTokenizer(vocab_file=filepath, do_lower_case=True)

    # tokenizer = tokenization.FullTokenizer(vocab_file="/home/jugs/Desktop/BERT-Pretrained/uncased_L-12_H-768_A-12/1/vocab.txt", do_lower_case=True) # vocab_file=args.vocab_file,

    token_example = tokenizer.tokenize(query)

    tokens = []
    tokens.append("[CLS]")
    segment_ids = []
    segment_ids.append(0)
    for token in token_example:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append('[SEP]')
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    max_seq_length = 256
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_id = 0

    instances = [{"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_id}]

    data = json.dumps({"signature_name": "serving_default", "instances": instances})

    response = requests.post(endpoints, data=data, headers=headers)
    prediction = json.loads(response.text)['predictions']
    idx_res = np.argmax(prediction)

    with urllib.request.urlopen("https://fashion-demo-assets.s3-ap-southeast-1.amazonaws.com/catDict.txt") as f:
      data = f.readlines()

    catlib = {}
    for l in data:
      strsplit = (l.decode('utf-8')).split('\t')
      catlib[strsplit[1].strip()] = strsplit[0].strip()

    #with open("/home/jugs/PycharmProjects/ExperimentalProjects/new_chic/catDict.txt", 'r') as fp:   # args.class_label,
    #    data = fp.readlines()

    #classLblDict = {}
    #for line in data:
    #    linesplit = line.split('\t')
    #    classLblDict[linesplit[1].strip()] = linesplit[0].strip()

    result = catlib[str(idx_res)]
    return result


# print(predict("swim wear for the summer"))
