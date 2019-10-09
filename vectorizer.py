import csv
import pickle
import re

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

twitter_modelfile = 'Models/word2vec_twitter_model.bin'
google_modelfile = 'Models/GoogleNews-vectors-negative300.bin'
twitter_cbow_gen = 'Models/Generated/W2V_cbow_400.txt'
twitter_skipgram_gen = 'Models/Generated/W2V_skipgram_400.txt'
wiki_word = 'Models/enwiki_20180420_win10_300d.txt'

glove_embeddings = {
    25: 'Models/glove.twitter.27B.25d.txt',
    50: 'Models/glove.twitter.27B.50d.txt',
    100: 'Models/glove.twitter.27B.100d.txt',
    200: 'Models/glove.twitter.27B.200d.txt',
    300: 'Models/glove.6B.300d.txt',
    400: 'Models/Generated/GloVe_400.txt',
}

glove_parsings = {
    25: 'Models/glove2vec.twitter.27B.25d.txt',
    50: 'Models/glove2vec.twitter.27B.50d.txt',
    100: 'Models/glove2vec.twitter.27B.100d.txt',
    200: 'Models/glove2vec.twitter.27B.200d.txt',
    300: 'Models/glove.6B.300d_word.txt',
    400: 'Models/Generated/GloVe2Vec_400.txt',
}


# Cleaning process to remove any punctuation, parentheses, question marks.
# This leaves only alphanumeric characters.
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")


class Word2VecModel():

    def __init__(self, train_dataset, test_dataset, class_qtd, base_model):
        self.classdataset = class_qtd
        self.word2vec_type = base_model

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)

        if base_model == 'CBOWGen':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_cbow_gen,
                encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'SkipGramGen':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_skipgram_gen,
                encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model == 'Wikipedia':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                wiki_word,
                encoding='utf-8'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter

        elif base_model == 'Twitter':
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                twitter_modelfile, binary=True, encoding='latin-1'
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter
        elif base_model in ['Google', 'random']:
            self.word2Vec_model = KeyedVectors.load_word2vec_format(
                google_modelfile, binary=True
            )
            self.dimension = self.word2Vec_model.vector_size
            self.tweet_length = 12  # 90 percentile value of number of words in a tweet based on Google
        print(f'The model has {self.dimension} dimensions')

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

        if self.word2vec_type in ['Twitter', 'Google', 'CBOWGen', 'SkipGramGen']:
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
                        values[i, index, :] = self.word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def random_vectorize(self, tweet_base):
        max_val = np.amax(self.word2Vec_model.syn0)
        min_val = np.amin(self.word2Vec_model.syn0)

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
        filename = 'Models/1_Word2Vec_' + self.word2vec_type +\
                    '_' + self.classdataset + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [train_vectors, test_vectors, train_labels, test_labels],
                f
            )


class GloveModel():

    def __init__(
        self,
        train_dataset,
        test_dataset,
        class_qtd,
        dimensions,
        translate=False
    ):
        # Necessário apenas se o modelo GloVe
        # não estiver "traduzido" para Word2Vec
        if translate:
            glove_embedding = glove_embeddings[dimensions]
            glove_parsing = glove_parsings[dimensions]
            _ = glove2word2vec(glove_embedding, glove_parsing)

        self.classdataset = class_qtd
        glove_file = glove_parsings[dimensions]
        self.gloVe_model = KeyedVectors.load_word2vec_format(
            glove_file,
            encoding='utf-8'
        )
        self.dimension = self.gloVe_model.vector_size

        self.train_tweets = self.parse_tweets(train_dataset)
        self.test_tweets = self.parse_tweets(test_dataset)
        self.tweet_length = 13  # 90 percentile value of number of words in a tweet based on Twitter

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

        print(f'{self.dimension} dimensions glove matrix has been created as'
              ' the input layer')

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
                        values[i, index, :] = self.gloVe_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        return values

    def save(self, train_vectors, train_labels, test_vectors, test_labels):
        filename = f'Models/1_GloVe_{self.dimension}_{self.classdataset}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(
                [train_vectors, test_vectors, train_labels, test_labels],
                f
            )

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
