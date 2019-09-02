import re

from gensim.models.keyedvectors import KeyedVectors


class Word2vec():
    def clean(self, sentence):
        remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
        return re.sub(remove_special_chars, "", sentence.lower())

    def train_vectorizer(self):
        self.Word2Vec_model = KeyedVectors.load_word2vec_format(
                                'word2vec_twitter_model.bin',
                                binary=True,
                                encoding='latin-1'
                              )

    def vectorize(self, sentence):
        pass
