from keras.layers import Dense, Reshape
from keras.models import Sequential

from vectorizer import Word2VecModel

word2vec = Word2VecModel(
    train_dataset='Data Collection/1_TrainingSet_3Class.csv',
    test_dataset='Data Collection/1_TestSet_3Class.csv',
    class_qtd='3class',
    base_model='Twitter'
)
train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()

vector_length = 13
vector_dimension = 400
target_dimension = 20

model = Sequential()
model.add(
    Reshape(
        (vector_length, vector_dimension, 1),
        input_shape=(vector_length, vector_dimension)
    ))
model.add(Dense(350, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(vector_dimension, activation='relu'))
