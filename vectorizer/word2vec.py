import csv
import pickle
import re

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

twitter_modelfile = 'Models/word2vec_twitter_model.bin'
google_modelfile = 'Models/GoogleNews-vectors-negative300.bin'

# Cleaning process to remove any punctuation, parentheses, question marks.
# This leaves only alphanumeric characters.
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")


class Word2VecModel():

    def __init__(self, train_dataset, test_dataset, class_qtd, base_model):
        self.classdataset = class_qtd
        self.word2vec_type = base_model  # 'Google' or 'Twitter', 'random'

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)

        if base_model == 'Twitter':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_modelfile, binary=True, encoding='latin-1'
            )
            self.dimension = self.Word2Vec_model.vector_size
            self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model in ['Google', 'random']:
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                google_modelfile, binary=True
            )
            self.dimension = self.Word2Vec_model.vector_size
            self.tweet_length = 12  # 90 percentile value of number of words in a tweet based on Google

    def clean(self, sentence):
        return re.sub(remove_special_chars, "", sentence.lower())

    def parse_tweets(self, filename):
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            tweets = []
            for tweet in reader:
                tweet[2] = self.clean(tweet[2])
                tweets.append(tweet)
        return tweets

    def vectorize(self):
        train_labels = [int(tweet[0]) for tweet in self.train_tweets]
        test_labels = [int(tweet[0]) for tweet in self.test_tweets]

        if self.word2vec_type in ["Twitter", 'Google']:
            vectorizer = CountVectorizer(
                min_df=1, stop_words='english',
                ngram_range=(1, 1),
                analyzer=u'word'
            )
            analyzer = vectorizer.build_analyzer()
            train_vectors = self.model_vectorize(
                tweet_base=self.train_tweets,
                analyzer=analyzer
            )
            test_vectors = self.model_vectorize(
                tweet_base=self.test_tweets,
                analyzer=analyzer
            )
        elif self.word2vec_type == 'random':
            train_vectors = self.random_vectorize(tweet_base=self.train_tweets)
            test_vectors = self.random_vectorize(tweet_base=self.test_tweets)

        print("{} word2vec matrix has been created as the input layer".format(
            self.word2vec_type
        ))

        return train_vectors, train_labels, test_vectors, test_labels

    def model_vectorize(self, tweet_base, analyzer):
        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )

        for i in range(len(tweet_base)):
            words_seq = analyzer(tweet_base[i][2])
            index = 0
            for word in words_seq:
                if index < self.tweet_length:
                    try:
                        values[i, index, :] = self.Word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def random_vectorize(self, tweet_base):
        max_val = np.amax(self.Word2Vec_model.syn0)
        min_val = np.amin(self.Word2Vec_model.syn0)

        values = np.zeros(
            (len(tweet_base), self.tweet_length, self.dimension),
            dtype=np.float32
        )
        for i in range(len(tweet_base)):
            values[i, :, :] = min_val +\
                              (max_val - min_val) * np.random.rand(
                                                        self.tweet_length,
                                                        self.dimension
                                                    )
        return values

    def save(self, train_vectors, train_labels, test_vectors, test_labels):
        filename = '1_Word2Vec_' + self.word2vec_type +\
                    '_' + self.classdataset + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [train_vectors, test_vectors, train_labels, test_labels],
                f
            )
