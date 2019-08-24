import keras
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class Language():
    """ Simple class to wrap information about a languae """
    def __init__(self, name, idxDict, vocabSize, maxSentLen):
        assert isinstance(name, str), 'name must have type str'
        assert isinstance(idxDict, dict), 'idxDict must have type dict'
        assert isinstance(vocabSize, int), 'vocbSize must have type int'
        assert isinstance(maxSentLen, int), 'maxSentLen must have type int'
        self.name = name
        self.idxDict = idxDict
        self.vocabSize = vocabSize
        self.maxSentLen = maxSentLen

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'NAME={self.name}/VOCAB_SIZE={self.vocabSize}/MAX_LEN={self.maxSentLen}'

    def add_word(self, newWord):
        """ Add a new word to the vocabulary """
        assert (newWord not in self.idxDict.keys()), f'{newWord} is already in the vocabulary'
        assert (self.vocabSize not in self.idxDict.values()), 'idx assignment error'
        self.idxDict.update({newWord : self.vocabSize})
        self.vocabSize += 1
        return True

    def remove_word(self, word):
        """ Remove a word from the dictionary """
        assert (word in self.idxDict.keys()), f'{word} is not in the vocabulary'
        del self.idxDict[word]
        self.vocabSize -= 1
        return True


def build_idx_dict(vocabSet):
    """ Builds dict mapping words from vocab iterable to unique id """
    return {word : i for i, word in enumerate(vocabSet)}


def split_and_clean_language_line(line, delimiter):
    # clean and lower line
    cleanLine = line.strip().lower()
    # separate between original language and translation
    original, translation = cleanLine.split(delimiter)
    return original, translation


def pad_sentence_ends(sentence, startToken='START', endToken='END'):
    """ Pads raw text sentence with start and end tokens """
    assert isinstance(sentence, str), 'sentence must have type str'
    return f'{startToken} {sentence} {endToken}'


def build_language_objects(filePath='fra-eng/fra.txt', featureLanguage,
                        targetLanguage, delimiter='\t', returnTrainData=True):
    """
    Builds dicts mapping words to unique int id and finds max line length
    in each language for padding one-hot vecs
    Args:
        filePath                     Path to the file containing feature and
                                        translated text
        featureLanguage             String of language for features
        targetLanguage              String of language for targets
        delimiter (*optional)       String of the delimiter separating
                                        features from targets defaults to tab
        returnTrainData             Whether to return training data lists of
                                        token lists for each sentence in
                                        features and targets
    Returns:
        Language() object storing the attributes of each language
    """
    # assertions
    SUPPORTED_LANGUAGES = ['english', 'french']
    assert (featureLanguage in SUPPORTED_LANGUAGES), f'{featureLanguage} is not supported'
    assert (targetLanguage in SUPPORTED_LANGUAGES), f'{targetLanguage} is not supported'
    assert isinstance(delimiter, str), 'delimiter must have type str'
    assert isinstance(returnTrainData, bool), 'returnTrainData must have type bool'
    # build set of all the words (and punctuation) in each language
    englishVocab, frenchVocab = set(), set()
    maxEnglish, maxFrench = 0, 0
    featureWords, targetWords = [], []
    with open(filePath, 'r') as translationFile:
        for line in tqdm(translationFile):
            # separate between english and french translation
            english, french = split_and_clean_language_line(line=line,
                                                            delimiter=delimiter)
            # tokenize words in each language
            englishLineWords = word_tokenize(english, language='english')
            frenchLineWords = word_tokenize(french, language='french')
            # update vocab sets
            for englishWord in englishLineWords:
                englishVocab.add(englishWord)
            for frenchWord in frenchLineWords:
                frenchVocab.add(frenchWord)
            # update max lengths
            maxEnglish  =   max(maxEnglish, len(englishLineWords))
            maxFrench   =   max(maxFrench, len(frenchLineWords))
            # update feature and target word lists
            featureWords.append(englishLineWords)
            targetWords.append(frenchLineWords)
    # build dict mapping each word to a unique int id
    englishIdxDict = build_idx_dict(englishVocab)
    frenchIdxDict = build_idx_dict(frenchVocab)
    # find vocab size of each language
    englishVocabSize = len(englishIdxDict)
    frenchVocabSize = len(frenchIdxDict)
    # convert language information into Language() objects
    englishObj = Language(name='english', idxDict=englishIdxDict,
                        vocabSize=englishVocabSize, maxSentLen=maxEnglish)
    frenchObj = Language(name='french', idxDict=frenchIdxDict,
                        vocabSize=frenchVocabSize, maxSentLen=maxFrench)
    if not returnTrainData:
        return englishObj, frenchObj
    else:
        return englishObj, frenchObj, featureWords, targetWords


def encode_training_data(featureWords, targetWords, featureLanguageObj,
                        targetLanguageObj, sampleCap=None):
    """
    Encodes matrix of raw unpadded feature and target words
    Args:
        featureWords:           List of token lists for each original sentence
        targetWords:            List of token lists for each target sentence
        featureLanguageObj      Language() object of feature language
        targetLanguageObj       Language() object of target language
        sampleCap (*optional)   Maximum number of samples to encode;
                                    defaults to None: all will be encoded
    """
    # assertions and formatting
    if not sampleCap:
        sampleCap = (len(featureWords) + 1)
    assert isinstance(sampleCap, int), 'sampleCap mut have type int'
    assert isinstance(featureLanguageObj, Language), 'featureLanguageObj must have type Language()'
    assert isinstance(targetLanguageObj, Language), 'targetLanguageObj must have type Language()'
    # get length of each sentence matrix in feature and target space
    featureSentLen  =   featureLanguageObj.maxSentLen
    targetSentLen   =   targetLanguageObj.maxSentLen
    # get lenght of one-hot vector in feature and target space
    featureVocabSize    =   featureLanguageObj.vocabSize
    targetVocabSize     =   targetLanguageObj.vocabSize
    # tuple of shapes for encoder inputs and decoder inputs and targets
    encoderInputShape = (sampleNum, featureSentLen, featureVocabSize)
    decoderInputShape = (sampleNum, targetSentLen, targetVocabSize)
    # initialize empty 3D arrays for training data
    encoderFeatures     =   np.zeros(shape=encoderInputShape, dtype='int.32')
    decoderFeatures     =   np.zeros(shape=decoderInputShape, dtype='int.32')
    decoderTargets      =   np.zeros(shape=decoderInputShape, dtype='int.32')
    # cache idx dict for each language
    featureIdxDict  =   featureLanguageObj.idxDict
    targetIdxDict   =   targetLanguageObj.idxDict
    # iterate over features and targets, building encoded arrays
    for sentNum, (featureSent, targetSent) in enumerate(zip(featureWords, targetWords)):
        # iterate over current feature sentence building 2D matrix of one-hot
        # encoded vectors of each word
        for wordNum, word in enumerate(featureSent):
            wordId = featureIdxDict[word]
            encoderFeatures[sentNum, wordNum, wordId] = 1
        # iterate over target sentence building one-hot matrix for decoder
        # inputs for teacher forcing and targets for decoder output advanced
        # one time step into the future
        for wordNum, word in enumerate(targetSent):




def encode_word(word, languageObj):
    """ Encodes a word as one-hot vector across vocab """
    assert (word in languageObj.idxDict), f'{word} in not in {languageObj.name}'
    wordVec = np.zeros(shape=(languageObj.vocabSize))
    wordVec[(languageObj.idxDict[word])] = 1
    return wordVec


def make_padding_vec(languageObj):
    emptyVec = np.zeros(shape=(languageObj.vocabSize))
    return emptyVec


def encode_sentence(sentence, language):
    """
    Encodes sentence in language as one-hot matrix of word vectors with padding
    up to language. Maxtix will have shape (vocabSize, maxSentLen)
    """
    # clean sentence
    cleanSentence = sentence.strip().lower()
    # tokenize sentence
    sentenceTokens = word_tokenize(cleanSentence, language=languageObj.name)
    # find size of padding to reach maxSentLen
    sentenceLength = len(sentenceTokens)
    paddingLength = languageObj.maxSentLen - sentenceLength
    assert (paddingLength >= 0), f'Sentence has length {sentenceLength}, but must be less than {languageObj.maxSentLen}.'
    # make one-hot encodings of words
    oneHots = [encode_word(word, languageObj) for word in sentenceTokens]
    # make empty padding vectors
    padding = [make_padding_vec(languageObj) for _ in range(paddingLength)]
    encodedMatrix = np.array(oneHots + padding)
    matrixShape = encodedMatrix.shape
    expectedDims = (languageObj.vocabSize, languageObj.maxSentLen)
    assert (matrixShape == expectedDims), f'Matrix has shape {matrixShape}, but should be {expectedDims}.'
    return encodedMatrix


def build_model():
    inputs = keras.layers.Input(shape=(MAX_LEN, VOCAB_SIZE))
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=300))(inputs)
    dense = keras.layers.Dense(units=1, activation='sigmoid')(lstm)
    model = keras.models.Model(inputs=inputs, outputs=dense)
    print(model.summary())
    return model


def build_encoder(latentDim=300):
    # encoder
    encoder_in = keras.layers.Input(shape=(None, VOCAB_SIZE), name='encoder_in')
    encoder_lstm = keras.layers.LSTM(units=latentDim, return_state=True, name='encoder_lstm')
    encoder_outputs, hidden_state, cell_state = encoder_lstm(encoder_in)
    encoder_states = [hidden_state, cell_state]
    # decoder
    decoder_in = keras.layers.Input(shape=(None, VOCAB_SIZE), name='decoder_in')
    decoder_lstm = keras.layers.LSTM(units=latentDim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_in, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(units=VOCAB_SIZE, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.models.Model([encoder_in, decoder_in], decoder_outputs)
    print(model.summary())
    return model


model = build_encoder()
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit([features, features], features, epochs=10, validation_split=0.1)


# model = build_model()
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(features, targets, epochs=100)
#
# while True:
#     sent = input('sent: ')
#     sentVec = np.expand_dims(encode_sent(sent), axis=0)
#     print(model.predict(sentVec))
