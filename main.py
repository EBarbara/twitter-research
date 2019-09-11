from vectorizer.word2vec import Word2VecModel


if __name__ == '__main__':
    word2vec = Word2VecModel(
        train_dataset='1_TrainingSet_3Class.csv',
        test_dataset='1_TestSet_3Class.csv',
        class_qtd='3class',
        base_model='Twitter'
    )
    train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()
    word2vec.save(train_vectors, train_labels, test_vectors, test_labels)
