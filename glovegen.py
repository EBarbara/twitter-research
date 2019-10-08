import csv
import re

from glove import Corpus, Glove
from nltk.tokenize import word_tokenize

# Cleaning process to remove any punctuation, parentheses, question marks.
# This leaves only alphanumeric characters.
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")


def clean(sentence):
    return re.sub(remove_special_chars, "", sentence.lower())


def parse_tweets(filename):
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        tweets = []
        for tweet in reader:
            tweet[2] = clean(tweet[2])
            tweets.append(tweet)
    return tweets


trainset = 'Data Collection/1_TrainingSet_3Class.csv'
testset = 'Data Collection/1_TrainingSet_3Class.csv'
components = 50

if __name__ == "__main__":
    print('Parsing tweets')
    dataset = parse_tweets(trainset) + parse_tweets(testset)
    tokens = []
    for data in dataset:
        temp = []
        for word in word_tokenize(data[2]):
            temp.append(word)
        tokens.append(temp)
    
    print('Generating GloVe model')
    corpus = Corpus()
    corpus.fit(tokens, window=10)

    glove = Glove(no_components=components, learning_rate=0.05)
 
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    # This is a shot in the dark... If the biases do 
    # something else besides network training, i'm screwed
    print('Saving model')
    index = len(glove.word_vectors)
    with open(f'Models/Generated/GloVe_{components}.txt', 'w') as f:
        for i in range(index):
            word = glove.inverse_dictionary[i]
            vector = glove.word_vectors[i]
            if len(vector) == components:
                f.write(f"{word} {' '.join(map(str, vector))}\n")
            else:
                print(f'Word {word} misvectored')
