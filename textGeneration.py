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
        vocabSize = max(reverseIdx)
    return textIds, wordIdx, reverseIdx, vocabSize


def one_hot_encode(wordId, vocabSize):
    oneHotVec = np.zeros(shape=(vocabSize, ))
    oneHotVec[wordId] = 1
    return oneHotVec


def vectorize_context_seq(idList, reverseIdx, vocabSize):
    """
    Vectorizes first seqSize tokens of idList using contextual attention
    and outputs feature matrix of token embeddings and one hot vector encoding
    target word id
    """
    assert (len(idList) == (SEQ_SIZE + 1)), f'Invalid idList length of {len(idList)}'
    seqWords = [reverseIdx[wordId] for wordId in idList[:-1]]
    vectorMatrix = bc.encode([seqWords], is_tokenized=True)[0]
    # remove cls and start
    filteredMatrix = vectorMatrix[1:-1]
    assert (len(filteredMatrix) == SEQ_SIZE), f'Invalid vector matrix shape of {vectorMatrix.shape}'
    targetId = idList[-1]
    targetVec = one_hot_encode(targetId, vocabSize)
    return filteredMatrix, targetVec


def generate_train_data(textIds, batchSize, reverseIdx, vocabSize):
    """ Generates batch for fitting """
    textLength = len(textIds)
    # pick random start point within the text
    startLoc = np.random.randint(0, (textLength - SEQ_SIZE))
    endLoc = startLoc + batchSize
    chunkSize = SEQ_SIZE + 1
    batchFeatures = []
    batchTargets = []
    for i in range(startLoc, endLoc):
        currentIds = textIds[i : (i + chunkSize)]
        filteredMatrix, targetVec = vectorize_context_seq(currentIds, reverseIdx, vocabSize)
        batchFeatures.append(filteredMatrix)
        batchTargets.append(targetVec)
    featureArray = np.array(batchFeatures)
    targetArray = np.array(batchTargets)
    yield(featureArray, targetArray)

textIds, wordIdx, reverseIdx, vocabSize = text_to_word_ids(TEXT_PATH)
batchSize = 10

# model
inputs = keras.layers.Input(shape=(SEQ_SIZE, 768))
lstm = keras.layers.LSTM(units=768, return_sequences=True)(inputs)
dense = keras.layers.Dense(units=vocabSize, activation='softmax')(lstm)
model = keras.models.Model(inputs=inputs, outputs=dense)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit_generator(generate_train_data(textIds, batchSize, reverseIdx, vocabSize), steps_per_epoch=10000, epochs=10)
