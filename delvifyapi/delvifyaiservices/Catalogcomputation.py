'''
 TODO: Requires BERT pre-trained model to host
 using bert-as-service to host/serve the BERT pre-trained model
 $ bert-serving-start -model_dir /home/jugs/Desktop/BERT-Pretrained/uncased_L-12_H-768_A-12/1/  -num_worker 1
'''

from bert_serving.client import BertClient
from numpy import dot
from numpy.linalg import norm
import operator
import json

# def computeSimilarityScore(queryEmbededing):
#     '''
#     Takes the input parameters as the sentence embedding that was returned from the
#     bert_client which was hosted from the server. And, uses the globally declared
#     variable (list of sentence embedding). Then generates the list of scores by
#     computing the similarity between user input and corpus.
#     :param queryEmbededing:
#     :return: list of score with user query and data corpus.
#     '''
#     # global item_sentenceEmbedding
#     scores = dict()
#     # queryEmbededing = np.transpose(queryEmbededing)
#     for key, value in item_sentenceEmbedding.items():
#         score_val = cos_sim(queryEmbededing[0], value)
#         scores[key] = score_val
#     return dict(sorted(scores.items(), reverse=True, key=operator.itemgetter(1)))


def cos_sim(a, b):
    '''
    Takes in two numpy array as it's parameter,
    and returns the cosine similarity of two vector(numpy)
    :param a: numpy array
    :param b: numpy array
    :return: cosine similarity score (scalar quantity) with a given equation
    '''
    return dot(a, b)/(norm(a)*norm(b))


def generateEmbedding(item_dictionary):
    bc = BertClient(check_length=False)
    item_text = list(item_dictionary.values())
    item_embedding = bc.encode(item_text)
    item_sku_embed = {}
    i = 0
    for key, value in item_dictionary.items():
        item_sku_embed[key] = item_embedding[i].tolist()
        i = i+1
    return item_sku_embed


def writeEmbedFile(item_sku_embedding, filePath, embedFilename):
    filePath = filePath.split('/')
    filePath = filePath[:-1]
    filePath = '/'.join(l for l in filePath)
    filePath = filePath + '/' + embedFilename
    with open(filePath, 'w') as ef:
        json.dump(item_sku_embedding, ef)


def writeSimilarSkus(scoresSku, filePath, similarSkuFilename):
    filePath = filePath.split('/')
    filePath = filePath[:-1]
    filePath = '/'.join(l for l in filePath)
    filePath = filePath + '/' + similarSkuFilename
    with open(filePath, 'w') as ef:
        json.dump(scoresSku, ef)


def computeSimilaritySku(item_sku_embedding, filepath, similarSkuFilename):
    scoresSku = {}
    for key, val in item_sku_embedding.items():
        top_scores = []
        top_skus = []
        top_score_skus = {}
        for k, v in item_sku_embedding.items():
            if k != key:
                score = cos_sim(val, v)
                if len(top_score_skus) >= 10:
                    tmp_top_sku_score = sorted(top_score_skus.items(), reverse=False, key=operator.itemgetter(1))
                    min_score = tmp_top_sku_score[0][1]
                    min_sku = tmp_top_sku_score[0][0]
                    if score > min_score:
                        top_skus.append(k)
                        top_skus.remove(min_sku)
                        top_score_skus[k] = score
                        top_score_skus.pop(min_sku)
                else:
                    top_score_skus[k] = score
                    top_scores.append(score)
                    top_skus.append(k)
        scoresSku[key] = top_skus
    writeSimilarSkus(scoresSku, filepath, similarSkuFilename)


def read_catalog(filePath, embedFileName, similarSkuFilename):
    item_dictionary = {}
    with open(filePath, 'r') as f:
        data = json.load(f)
    for line in data:
        sku = line['SKU']
        title = line["Name"]
        category = line["Category"]
        if len(line["Description"]) >= 50:
            description = line["Description"]
        else:
            description = ""
        productUrl = line["OriginalUrl"]
        imageUrl = line["Image"]
        price = line["Price"]
        currency = line["Currency"]
        others = line["Brand"]
        item_dictionary[sku] = title + category + description + others
    item_sku_embedding = generateEmbedding(item_dictionary)
    writeEmbedFile(item_sku_embedding, filePath, embedFileName)
    computeSimilaritySku(item_sku_embedding, filePath, similarSkuFilename)


# read_catalog("/home/jugs/PycharmProjects/DelvifyWebAPI/delvifyWebServices/5555/catalog.json", "myembeded.txt", "mysimilar.txt")