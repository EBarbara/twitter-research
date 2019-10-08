import csv
import re

from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec

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
components = 100

if __name__ == "__main__":
    print('Parsing tweets')
    dataset = parse_tweets(trainset) + parse_tweets(testset)
    tokens = []
    for data in dataset:
        temp = []
        for word in word_tokenize(data[2]):
            temp.append(word)
        tokens.append(temp)

    print('Generating W2V - cbow model')
    cbow_model = Word2Vec(tokens, min_count=1, size=components)
    print('Saving model')
    cbow_model.wv.save_word2vec_format(f'Models/Generated/W2V_cbow_{components}.txt')

    print('Generating W2V - skipgram model')
    skipgram_model = Word2Vec(tokens, min_count=1, sg=1, size=components)
    print('Saving model')
    skipgram_model.wv.save_word2vec_format(f'Models/Generated/W2V_skipgram_{components}.txt')
