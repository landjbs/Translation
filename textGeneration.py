import keras
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from bert_serving.client import BertClient

SEQ_SIZE = 100
BERT_PATH = '/Users/landonsmith/Desktop/shortBert'

# configure BERT client
bc = BertClient(check_length=True)
bertLaunch = f'bert-serving-start -model_dir {BERT_PATH} -num_worker=3 -max_seq_len={SEQ_SIZE} -pooling-strategy=NONE'
# bert-serving-start -model_dir /Users/landonsmith/Desktop/shortBert -num_worker=3 -max_seq_len=100 -pooling-strategy=NONE

def text_to_word_ids(textPath):
    lenList = []
    wordIdx = dict()
    with open(textPath, 'r') as textFile:
        rawText = textFile.read()
        cleanText = rawText.lower()
        textWords = word_tokenize(cleanText, language='english')
        wordIdx = {word : i for i, word in enumerate(textWords)}
        reverseIdx = {i : word for word, i in wordIdx.items()}
        word_to_id = lambda word : wordIdx[word]
        textIds = list(map(word_to_id, textWords))
    return textIds, wordIdx, reverseIdx


def encode
