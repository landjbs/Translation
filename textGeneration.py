import keras
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from bert_serving.client import BertClient

SEQ_SIZE = 100
BERT_PATH = '/Users/landonsmith/Desktop/shortBert'
TEXT_PATH = 'gatsby.txt'


# configure BERT client
bc = BertClient(check_length=True)
bertLaunch = f'bert-serving-start -model_dir {BERT_PATH} -num_worker=3 -max_seq_len={SEQ_SIZE} -pooling-strategy=NONE'
# bert-serving-start -model_dir /Users/landonsmith/Desktop/shortBert -num_worker=3 -max_seq_len=100 -pooling-strategy=NONE


def text_to_word_ids(textPath):
    """
    Builds list of word ids for each word in text, encoded with wordIdx and
    decipherable with reverseIdx
    """
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

textIds, wordIdx, reverseIdx = text_to_word_ids()

def vectorize_context_seq(idList, reverseIdx):
    seqString = ' '.join(reverseIdx[wordId] for wordId in idList)
    print(seqString)

vectorize_context_seq(['hi', 'how', 'are', 'you'], reverseIdx)
