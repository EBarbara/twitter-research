from vectorizer.word2vec import Word2VecModel

import deep_learning
from deep_learning import ConvolutedNeuralNetwork

if __name__ == '__main__':
    word2vec = Word2VecModel(
        train_dataset='Data Collection/1_TrainingSet_3Class.csv',
        test_dataset='Data Collection/1_TestSet_3Class.csv',
        class_qtd='3class',
        base_model='Twitter'
    )
    train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()
    word2vec.save(train_vectors, train_labels, test_vectors, test_labels)

    cnn = ConvolutedNeuralNetwork(
        train_vectors=train_vectors,
        train_labels=train_labels
    )
    deep_learning.train_network(
        network=cnn,
        test_vectors=test_vectors,
        test_labels=test_labels,
        weight_filepath='Models/weights.best2.hdf5',
        hist_filepath='Models/Accuracy-History-CNN-2class.pickle'
    )
    deep_learning.test_network(
        network=cnn,
        weight_filepath='Models/weights.best2.hdf5',
        loss_function='categorical_crossentropy',
        optimizer_function='adam',
        metrics=['accuracy'],
        test_vectors=test_vectors,
        test_labels=test_labels
    )
