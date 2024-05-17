import random
import json
import pickle
import numpy as np
import tensorflow as tf
import keras

import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer

intents = json.loads(open('intense.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intents['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList,intent['tag']))
        if intent['tag'] not in classes
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if not any(letter in ignoreLetters for letter in word)]
words = sorted(set(classes))

classes=sorted(set(classes))

pickle.dump(words,open('words.pk1','wb'))
pickle.dump(classes,open('Classes.pk1','wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag=[]
    wordPatterns = documents[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower().strip()) for words in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)

    outputrow = list(outputEmpty)
    outputrow[classes.indexdocument[1]] = 1
    training.append(bag + outputrow)

random.shuffle(training)
training  = np.append(training)

trainX = training[:, :len(words)]
trainY = training[:, :len(words):]

model = keras.Sequential()

model.add(keras.layers.Dense(128,))




# strip








