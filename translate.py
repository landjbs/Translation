import keras
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

FRENCH_PATH = 'fra-eng/fra.txt'

class Language():
    """ Simple class to wrap information about a language """

    def __init__(self, name, idxDict, vocabSize, maxSentLen, reverese=True):
        assert isinstance(name, str), 'name must have type str'
        assert isinstance(idxDict, dict), 'idxDict must have type dict'
        assert isinstance(vocabSize, int), 'vocbSize must have type int'
        assert isinstance(maxSentLen, int), 'maxSentLen must have type int'
        self.name = name
        self.idxDict = idxDict
        self.vocabSize = vocabSize
        self.maxSentLen = maxSentLen
        if reverese:
            self.reverseIdx = self.reverseIdx()
        else:
            self.reverseIdx = None

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

    def reverse_idx(self):
        """ Builds reverse idx mapping ids to words """
        self.reverseIdx = {word : id for id, word in self.idxDict.items()}


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


def build_language_objects(filePath, featureLanguage, targetLanguage,
                            delimiter='\t', returnTrainData=True):
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
        for i, line in enumerate(tqdm(translationFile)):
            if i > 1000:
                break
            # separate between english and french translation
            featureSent, targetSent = split_and_clean_language_line(line=line,
                                                            delimiter=delimiter)
            # pad sentence beginnings and ends
            paddedFeatureSent = pad_sentence_ends(featureSent)
            paddedTargetSent = pad_sentence_ends(targetSent)
            # tokenize words in each language
            englishLineWords = word_tokenize(text=paddedFeatureSent,
                                            language=featureLanguage)
            frenchLineWords = word_tokenize(text=paddedTargetSent,
                                            language=targetLanguage)
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
                        targetLanguageObj, sampleNum=None):
    """
    Encodes matrix of raw unpadded feature and target words
    Args:
        featureWords:           List of token lists for each original sentence
        targetWords:            List of token lists for each target sentence
        featureLanguageObj:     Language() object of feature language
        targetLanguageObj:      Language() object of target language
        sampleNum (*optional):  Maximum number of samples to encode;
                                    defaults to None: all will be encoded
    Returns:
        encoderFeatures:        Numpy matrix of one-hot encoded words for each
                                    sentence padded to maxSentLen. Used to train
                                    encoder LSTM cell and hidden states. Shape
                                    is (sampleNum, featureSentLen,
                                    featureVocabSize)
        decoderFeatures:       Numpy matrix of one-hot encoded words for each
                                    sentence padded to maxSentLen. Used as
                                    input to decoder LSTM for teacher forcing
                                    training speed improvement. Shape is
                                    (sampleNum, targetSentLen, targetVocabSize)
        decoderTargets:        Numpy matrix of one-hot encoded words advanced
                                    by one time step with respect to
                                    decoderFeatures. Used as final prediction
                                    target for decoder LSTM (and model).
    """
    # assertions and formatting
    if not sampleNum:
        sampleNum = (len(featureWords) + 1)
    assert isinstance(sampleNum, int), 'sampleNum mut have type int'
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
    encoderFeatures     =   np.zeros(shape=encoderInputShape, dtype='int32')
    decoderFeatures     =   np.zeros(shape=decoderInputShape, dtype='int32')
    decoderTargets      =   np.zeros(shape=decoderInputShape, dtype='int32')
    # cache idx dict for each language
    featureIdxDict  =   featureLanguageObj.idxDict
    targetIdxDict   =   targetLanguageObj.idxDict
    # iterate over features and targets, building encoded arrays
    for sentNum, (featureSent, targetSent) in tqdm(enumerate(zip(featureWords,
                                                                targetWords))):
        if sentNum >= sampleNum:
            break
        # iterate over current feature sentence building 2D matrix of one-hot
        # encoded vectors of each word
        for wordNum, word in enumerate(featureSent):
            wordId = featureIdxDict[word]
            encoderFeatures[sentNum, wordNum, wordId] = 1
        # iterate over target sentence building one-hot matrix for decoder
        # inputs for teacher forcing and targets for decoder output advanced
        # one time step into the future
        for wordNum, word in enumerate(targetSent):
            wordId = targetIdxDict[word]
            decoderFeatures[sentNum, wordNum, wordId] = 1
            # decoder target will be the same word but one time-step ahead
            if wordNum > 0:
                decoderTargets[sentNum, (wordNum - 1), wordId] = 1
    return encoderFeatures, decoderFeatures, decoderTargets


def build_encoder_decoder(featureLanguageObj, targetLanguageObj, latentDims=300):
    """
    Builds encoder/decoder LSTM model with final dense layer softmax predictions
    of next word. Uses LSTM encoder to generate cell and hidden state vector
    to initialize decoder LSTM. Uses teacher forcing to avoid model instability
    and slow training in decoder.
    Args:
        featureLanguageObj:         Language() obj for feature data
        targetLanguageObj:          Language() obj for target data
        latentDims:                 Number of latent dimensions for encoder
                                        LSTM (aka. length of hidden and
                                        cell vector) and initial state of
                                        decoder LSTM.
    Returns:
        Non-compiled model of encoder/decoder LSTM.
    """
    # cache language info from Language() objects
    featureVocabSize = featureLanguageObj.vocabSize
    targetVocabSize = targetLanguageObj.vocabSize
    ## encoder architecture ##
    # encoder takes one-hot vector of input word token
    encoder_in = keras.layers.Input(shape=(None, featureVocabSize),
                                    name='encoder_in')
    # LSTM builds cell vector of size latentDims from inputs
    encoder_lstm = keras.layers.LSTM(units=latentDims, return_state=True,
                                    name='encoder_lstm')
    encoder_outputs, hidden_state, cell_state = encoder_lstm(encoder_in)
    # pull just the hidden and cell state from the lstm
    encoder_states = [hidden_state, cell_state]
    ## decoder architecture ##
    # decoder takes one-hot vector of correct word token (teach forcing)
    decoder_in = keras.layers.Input(shape=(None, targetVocabSize),
                                    name='decoder_in')
    # LSTM builds cell vector of size latentDims from inputs and encoder states
    decoder_lstm = keras.layers.LSTM(units=latentDims, return_sequences=True,
                                    return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_in,
                                        initial_state=encoder_states)
    # dense layer uses softmax activation for token prediction
    decoder_dense = keras.layers.Dense(units=targetVocabSize,
                                        activation='softmax',
                                        name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    # build dictionary of modle layers for using later
    modelDict = {'encoder_in':encoder_in, 'encoder_states':encoder_states,
                }
    # model takes encoder and decoder inputs and predicts on decoder outputs
    model = keras.models.Model([encoder_in, decoder_in], decoder_outputs)
    return model


# gather the data from file
(englishObj,
    frenchObj,
    featureWords,
    targetWords
) = build_language_objects(filePath=FRENCH_PATH, featureLanguage='english',
                        targetLanguage='french', returnTrainData=True)
# encode the data
(encoderFeatures,
    decoderFeatures,
    decoderTargets
) = encode_training_data(featureWords=featureWords, targetWords=targetWords,
                        featureLanguageObj=englishObj,
                        targetLanguageObj=frenchObj, sampleNum=1000)

# build encoder/decoder model
# model = build_encoder_decoder(featureLanguageObj=englishObj,
#                                     targetLanguageObj=frenchObj)

featureLanguageObj = englishObj
targetLanguageObj = frenchObj
latentDims = 300
# cache language info from Language() objects
featureVocabSize = featureLanguageObj.vocabSize
targetVocabSize = targetLanguageObj.vocabSize
## encoder architecture ##
# encoder takes one-hot vector of input word token
encoder_in = keras.layers.Input(shape=(None, featureVocabSize),
                                name='encoder_in')
# LSTM builds cell vector of size latentDims from inputs
encoder_lstm = keras.layers.LSTM(units=latentDims, return_state=True,
                                name='encoder_lstm')
encoder_outputs, hidden_state, cell_state = encoder_lstm(encoder_in)
# pull just the hidden and cell state from the lstm
encoder_states = [hidden_state, cell_state]
## decoder architecture ##
# decoder takes one-hot vector of correct word token (teach forcing)
decoder_in = keras.layers.Input(shape=(None, targetVocabSize),
                                name='decoder_in')
# LSTM builds cell vector of size latentDims from inputs and encoder states
decoder_lstm = keras.layers.LSTM(units=latentDims, return_sequences=True,
                                return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_in,
                                    initial_state=encoder_states)
# dense layer uses softmax activation for token prediction
decoder_dense = keras.layers.Dense(units=targetVocabSize,
                                    activation='softmax',
                                    name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
# build dictionary of modle layers for using later
modelDict = {'encoder_in':encoder_in, 'encoder_states':encoder_states,
            }
# model takes encoder and decoder inputs and predicts on decoder outputs
model = keras.models.Model([encoder_in, decoder_in], decoder_outputs)


# compile model
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# fit the model
model.fit([encoderFeatures, decoderFeatures], decoderTargets, epochs=1, validation_split=0.1)

## Sampling ##
encoder_model = keras.models.Model(encoder_in, encoder_states)

from keras.layers import Input


decoder_state_input_h = Input(shape=(latentDims,))
decoder_state_input_c = Input(shape=(latentDims,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_in, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.models.Model(
    [decoder_in] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
