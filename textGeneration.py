import keras
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def text_to_words(textPath):
    lenList = []
    with open(textPath, 'r') as textFile:
        rawText = textFile.read()
        cleanText = rawText.lower()
        textWords = word_tokenize(cleanText, language='english')
        


text_to_words('gatsby.txt')
