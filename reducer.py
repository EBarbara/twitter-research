from keras.layers import Dense, Input
from keras.models import Model

from vectorizer import Word2VecModel

word2vec = Word2VecModel(
    train_dataset='Data Collection/1_TrainingSet_3Class.csv',
    test_dataset='Data Collection/1_TestSet_3Class.csv',
    class_qtd='3class',
    base_model='Twitter'
)
train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()

vector_length = 13
n_dim = 400
enc_dim = 20

input = Input(shape=(n_dim, ))
encode1 = Dense(350, activation='relu')(input)
encode2 = Dense(300, activation='relu')(encode1)
encode3 = Dense(250, activation='relu')(encode2)
encode4 = Dense(200, activation='relu')(encode3)
encode5 = Dense(150, activation='relu')(encode4)
encode6 = Dense(100, activation='relu')(encode5)
encode7 = Dense(50, activation='relu')(encode6)
encode8 = Dense(enc_dim, activation='relu')(encode7)
decode1 = Dense(50, activation='relu')(encode8)
decode2 = Dense(100, activation='relu')(decode1)
decode3 = Dense(100, activation='relu')(decode2)
decode4 = Dense(100, activation='relu')(decode3)
decode5 = Dense(100, activation='relu')(decode4)
decode6 = Dense(100, activation='relu')(decode5)
decode7 = Dense(100, activation='relu')(decode6)
decode8 = Dense(n_dim, activation='sigmoid')(decode7)

model = Model(input=input, output=decode8)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(
    train_vectors,
    train_vectors,
    nb_epoch=20,
    batch_size=100,
    shuffle=True,
    validation_data=(test_vectors, test_vectors)
)

encoder = Model(input=input, output=encode8)
encode_input = Input(shape=(enc_dim, ))  # WTF??
trainer_output = encoder.predict(train_vectors)
tester_output = encoder.predict(test_vectors)

print(f'the output, {tester_output} is a {type(tester_output)}')
