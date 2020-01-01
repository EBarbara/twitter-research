import time

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA

modelfile = 'E:/Models/word2vec_twitter_model.bin'

n_dim = 400
enc_dim = 200

words = set()

if __name__ == '__main__':
    print('loading pretrained model')
    word2Vec_model = KeyedVectors.load_word2vec_format(
        modelfile,
        binary=True,
        encoding='latin1'
    )

    vectors = []
    words = []

    for index, word in enumerate(word2Vec_model.vocab):
        vectors.append(word2Vec_model[word])
        words.append(word)

    vectors_np = np.asarray(vectors)
    print(f'Run at {time.process_time()} seconds')

    print('PCA to get Top Components')
    pca_top = PCA(n_components=n_dim)
    vectors_np = vectors_np - np.mean(vectors_np)
    vectors_fit = pca_top.fit_transform(vectors_np)
    U1 = pca_top.components_
    print(f'Run at {time.process_time()} seconds')

    print('Removing Projections on Top Components')
    z = []
    for i, x in enumerate(vectors_np):
        for u in U1[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)
    print(f'Run at {time.process_time()} seconds')

    print('PCA Dim Reduction')
    pca_reduce = PCA(n_components=enc_dim)
    vectors_reduce = z - np.mean(z)
    vectors_reduced = pca_reduce.fit_transform(vectors_reduce)
    print(f'Run at {time.process_time()} seconds')

    print('PCA to do Post-Processing Again')
    pca_post = PCA(n_components=enc_dim)
    vectors_post = vectors_reduced - np.mean(vectors_reduced)
    vectors_post = pca_post.fit_transform(vectors_post)
    Ufit = pca_post.components_

    vectors_reduced = vectors_reduced - np.mean(vectors_reduced)
    print(f'Run at {time.process_time()} seconds')

    print('Saving embeddings')
    final_pca_embeddings = {}
    with open(
        f'E:/Models/Generated/pca_embed_{enc_dim}.txt',
        'w',
        encoding="utf8"
    ) as output:
        output.write(f'{len(words)} {enc_dim}\n')
        for i, x in enumerate(words):
            final_pca_embeddings[x] = vectors_reduced[i]
            output.write(f'{x} ')
            for u in Ufit[0:7]:
                final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(
                    u.transpose(), final_pca_embeddings[x]
                ) * u

            for term in final_pca_embeddings[x]:
                stringed = format(term, 'f')
                output.write(f'{stringed} ')

            output.write('\n')
    print(f'Run at {time.process_time()} seconds')
