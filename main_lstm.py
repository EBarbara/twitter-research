from vectorizer import Word2VecModel


if __name__ == '__main__':
    word2vec = Word2VecModel(
        train_dataset='Data Collection/1_TrainingSet_2Class.csv',
        test_dataset='Data Collection/1_TestSet_2Class.csv',
        class_qtd='2class',
        base_model='Google'
    )
    train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()
    word2vec.save(train_vectors, train_labels, test_vectors, test_labels)
