import csv
import re
import time

from gensim.models.keyedvectors import KeyedVectors
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
    print('building vectorizer')
    vectorizer = CountVectorizer(
        min_df=1,
        stop_words='english',
        ngram_range=(1, 1),
        analyzer=u'word'
    )
    analyzer = vectorizer.build_analyzer()
    print(f'Run at {time.process_time()} seconds')

    print('loading training dataset')
    train_tweets = parse_tweets(train_dataset)
    for i in range(len(train_tweets)):
        words_seq = analyzer(train_tweets[i][2])
        for word in words_seq:
            words.add(word)
    print(f'Run at {time.process_time()} seconds')

    print('loading testing dataset')
    test_tweets = parse_tweets(test_dataset)
    for i in range(len(test_tweets)):
        words_seq = analyzer(test_tweets[i][2])
        for word in words_seq:
            words.add(word)
    print(f'Run at {time.process_time()} seconds')

    print('loading pretrained model')
    wordlist = list(words)
    word2Vec_model = KeyedVectors.load_word2vec_format(
        twitter_modelfile, binary=True, encoding='latin-1'
    )
    print(f'Run at {time.process_time()} seconds')

    print('building zeroed vector')
    wordvectors = np.zeros(
        (len(wordlist), 400),
        dtype=np.float32
    )
    print(f'Run at {time.process_time()} seconds')

    print('loading word embeddings on vector')
    for i in range(len(wordlist)):
        word = wordlist[i]
        try:
            wordvectors[i, :] = word2Vec_model[word]
        except KeyError:
            pass
    print(f'Run at {time.process_time()} seconds')

    print('Modelling Autoencoder')
    input = Input(shape=(n_dim, ))
    encode1 = Dense(300, activation='relu')(input)
    encode2 = Dense(200, activation='relu')(encode1)
    encode3 = Dense(100, activation='relu')(encode2)
    encode4 = Dense(50, activation='relu')(encode3)
    encode5 = Dense(enc_dim, activation='relu')(encode4)
    decode1 = Dense(50, activation='relu')(encode5)
    decode2 = Dense(100, activation='relu')(decode1)
    decode3 = Dense(200, activation='relu')(decode2)
    decode4 = Dense(300, activation='relu')(decode3)
    decode5 = Dense(n_dim, activation='relu')(decode4)
    model = Model(inputs=input, outputs=decode5)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f'Run at {time.process_time()} seconds')

    print('Training autoencoder')
    model.fit(
        wordvectors,
        wordvectors,
        epochs=20,
        batch_size=100,
        shuffle=True
    )
    print(f'Run at {time.process_time()} seconds')

    print('Encoding vectors')
    encoder = Model(input=input, output=encode5)
    encode_input = Input(shape=(enc_dim, ))  # WTF??
    encode_words = encoder.predict(wordvectors)
    print(f'Run at {time.process_time()} seconds')

    print('Saving model')
    index = len(wordlist)
    with open(f'Models/Generated/Encoded_Twitter_W2Vec_20.txt', 'w') as f:
        for i in range(index):
            word = wordlist[i]
            vector = encode_words[i]
            if len(vector) == enc_dim:
                f.write(f"{word} {' '.join(map(str, vector))}\n")
            else:
                print(f'Word {word} misvectored')
    print(f'Run at {time.process_time()} seconds')
