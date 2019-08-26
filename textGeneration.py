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
bertLaunch = f'bert-serving-start -model_dir {BERT_PATH} -num_worker=3 -max_seq_len={SEQ_SIZE + 2} -pooling_strategy=NONE'
# bert-serving-start -model_dir /Users/landonsmith/Desktop/shortBert -num_worker=3 -max_seq_len=102 -pooling_strategy=NONE


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
        wordIdx = {word : i for i, word in tqdm(enumerate(textWords))}
        reverseIdx = {i : word for word, i in tqdm(wordIdx.items())}
        word_to_id = lambda word : wordIdx[word]
        textIds = [word_to_id(word) for word in tqdm(textWords)]
    return textIds, wordIdx, reverseIdx


textIds, wordIdx, reverseIdx = text_to_word_ids(TEXT_PATH)


def vectorize_context_seq(idList, reverseIdx):
    """
    Vectorizes first seqSize tokens of idList using contextual attention
    and outputs feature matrix of token embeddings and target word id
    """
    assert (len(idList)==(SEQ_SIZE + 1))
    seqWords = [reverseIdx[wordId] for wordId in idList[:-1]]
    vectorMatrix = bc.encode([seqWords], is_tokenized=True)[0]
    targetId = idList[-1]
    return vectorMatrix, targetId

for i in range(0, 1000):
    _, x = vectorize_context_seq(textIds[i:(i+101)], reverseIdx)
    print(reverseIdx[x])
