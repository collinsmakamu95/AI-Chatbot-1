import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("human.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []
# Stemming
# this is to loop all the tags in the json  file#
for human in data["human"]:
    for pattern in human["patterns"]:
        wds = nltk.word_tokenize(pattern)
        words.extend(wds)
        docs_x.append(wds)
        docs_y.append(human["tag"])
        if human["tag"] not in labels:
            labels.append(human["tag"])

# this is convert all the converted words into lowercase to avoid mix-up
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# training and testing the ai

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Building the  ai model#
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
# 8 neurons fully connected
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# parsing all the training data 1000 times the more it sees the data the more accurate

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


# getting the ai to interact making predictions


def bag_of_words(s, word_s):
    bag = [0 for _ in range(len(word_s))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(word_s):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


# asking the user for interaction
# argmax gives the greatest index in the probability bag
# train the bot to know when to recognize an unrecognized response

def chat():
    print("Systems up and running the conversation can begin!   (Type quit to end session)")
    while True:
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:

            for tg in data["human"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I'm sorry I didn't quite get that. Try again")


chat()
