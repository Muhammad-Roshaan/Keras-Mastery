import random
import json
import pickle
import numpy as np
import tensorflow as tf
import keras

import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('D:\Python\Chatbot\Chatbots\Virtual Environment (Colab)\intense.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:  # Corrected variable name
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if not any(letter in ignoreLetters for letter in word)]
words = sorted(set(words))  # Corrected variable name

pickle.dump(words, open('words.pkl', 'wb'))  # Corrected file extension
pickle.dump(classes, open('Classes.pkl', 'wb'))  # Corrected file extension

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]  # Corrected variable name
    wordPatterns = [lemmatizer.lemmatize(word.lower().strip()) for word in wordPatterns]  # Corrected variable name
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)  # Corrected indentation

    outputrow = list(outputEmpty)
    outputrow[classes.index(document[1])] = 1
    training.append(bag + outputrow)

random.shuffle(training)
training = np.array(training)  # Corrected conversion to numpy array

trainX = training[:, :len(words)]
trainY = training[:, len(words):]  # Corrected slicing indices

model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_simplilearnmodel.h5')  # Removed 'hist'
print("Executed")
