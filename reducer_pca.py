from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train_dataset = 'Data Collection/1_TrainingSet_3Class.csv'
test_dataset = 'Data Collection/1_TestSet_3Class.csv'
twitter_modelfile = 'Models/word2vec_twitter_model.bin'
remove_special_chars = re.compile("[^A-Za-z0-9 ]+")

n_dim = 400
enc_dim = 20

words = set()


def parse_tweets(filename):
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        tweets = []
        for tweet in reader:
            tweet[2] = re.sub(remove_special_chars, "", tweet[2].lower())
            tweets.append(tweet)
    return tweets


if __name__ == '__main__':
    # Load Vectors
    # PCA to get Top Components
    # Removing Projections on Top Components
    # PCA Dim Reduction
    # PCA to do Post-Processing Again
