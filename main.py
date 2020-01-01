# Variáveis usadas
inputs = {
    '2_class': (
        'Data Collection/1_TrainingSet_2Class.csv',
        'Data Collection/1_TestSet_2Class.csv'
    ),
    '3_class': (
        'Data Collection/1_TrainingSet_3Class.csv',
        'Data Collection/1_TestSet_3Class.csv'
    ),
}

vector_models = {
    'word2vec_400_twitter': 'E:/Models/word2vec_twitter_model.bin',
    'word2vec_400_twitter_pca_200': 'E:/Models/Generated/pca_embed_200.txt',
    'word2vec_400_twitter_pca_100': 'E:/Models/Generated/pca_embed_100.txt',
    'word2vec_400_twitter_pca_50': 'E:/Models/Generated/pca_embed_50.txt',
    'word2vec_400_twitter_pca_20': 'E:/Models/Generated/pca_embed_20.txt',
    'word2vec_300_googlenews': 'E:/Models/GoogleNews-vectors-negative300.bin',
    'glove_200_twitter': 'E:/Models/glove2vec.twitter.27B.200d.txt',
    'glove_200_twitter_pca_100': 'E:/Models/Generated/pca_embed_glove_200_100.txt',
    'glove_200_twitter_pca_20': 'E:/Models/Generated/pca_embed_glove_200_20.txt',
    'glove_100_twitter': 'E:/Models/glove2vec.twitter.27B.100d.txt',
    'glove_100_twitter_pca_50': 'E:/Models/Generated/pca_embed_glove_100_50.txt',
    'glove_100_twitter_pca_20': 'E:/Models/Generated/pca_embed_glove_100_20.txt',
    'glove_50_twitter': 'E:/Models/glove2vec.twitter.27B.50d.txt',
    'glove_50_twitter_pca_25': 'E:/Models/Generated/pca_embed_glove_50_25.txt',
    'glove_50_twitter_pca_20': 'E:/Models/Generated/pca_embed_glove_50_20.txt',
    'glove_25_twitter': 'E:/Models/glove2vec.twitter.27B.25d.txt',
    'glove_25_twitter_pca_20': 'E:/Models/Generated/pca_embed_glove_25_20.txt',
}