import time

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA


class Vectors:

    def __init__(
        self,
        input_dim, output_dim, top_count,
        input_path, binary=False, encoding='utf8'
    ):
        print('loading pretrained model')
        self.word_list = []
        self.n_dim = input_dim
        self.enc_dim = output_dim
        self.top_count = top_count

        word2Vec_model = KeyedVectors.load_word2vec_format(
            input_path,
            binary=binary,
            encoding=encoding
        )

        vectors = []
        for index, word in enumerate(word2Vec_model.vocab):
            vectors.append(word2Vec_model[word])
            self.word_list.append(word)

        self.word_vectors = np.asarray(vectors)
        print(f'Run at {time.process_time()} seconds')

    def remove_projections(self, components):
        z = []
        for i, x in enumerate(self.word_vectors):
            for u in components[0:self.top_count]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)

    def normalize(self):
        self.word_vectors = self.word_vectors - np.mean(self.word_vectors)

    def get_pca_components(self):
        pca = PCA(n_components=self.n_dim)
        vectors_unmean = self.word_vectors - np.mean(self.word_vectors)
        pca.fit_transform(vectors_unmean)
        components = pca.components_
        print(f'Run at {time.process_time()} seconds')
        return components

    def pca_reduce(self):
        pca = PCA(n_components=self.enc_dim)
        vectors_unmean = self.word_vectors - np.mean(self.word_vectors)
        self.word_vectors = pca.fit_transform(vectors_unmean)
        self.n_dim = self.enc_dim
        print(f'Run at {time.process_time()} seconds')

    def save(self, output_path, components, encoding='utf8'):
        print('Saving embeddings')
        final_pca_embeddings = {}
        with open(output_path, 'w', encoding=encoding) as output:
            output.write(f'{len(self.word_list)} {self.enc_dim}\n')
            for i, x in enumerate(self.word_list):
                #print debug
                print(f'x = {x}, i = {i}; We have {len(self.word_list)} words and {len(self.word_vectors)} vectors')
                final_pca_embeddings[x] = self.word_vectors[i]
                output.write(f'{x} ')
                for u in components[0:7]:
                    final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(
                        u.transpose(), final_pca_embeddings[x]
                    ) * u

                for term in final_pca_embeddings[x]:
                    stringed = format(term, 'f')
                    output.write(f'{stringed} ')

                output.write('\n')
        print(f'Run at {time.process_time()} seconds')


if __name__ == '__main__':
    top_count = 7
    configs = [
        (300, 50,  'D:\\models\\glove_s300.txt', 'D:\\models\\glove_pca_normalize_300-50.txt'),
        (100, 50,  'D:\\models\\word2vec_s100.txt', 'D:\\models\\word2vec_pca_normalize_100-50.txt'),
        (100, 50,  'D:\\models\\fasttext_s100.txt', 'D:\\models\\fasttext_pca_normalize_100-50.txt'),
        (100, 50,  'D:\\models\\glove_s100.txt', 'D:\\models\\glove_pca_normalize_100-50.txt'),
    ]

    for config in configs:
        vectors = Vectors(
            input_dim=config[0],
            output_dim=config[1],
            top_count=top_count,
            input_path=config[2]
        )

        components = vectors.get_pca_components()
        vectors.remove_projections(components)
        vectors.pca_reduce()
        reduced_components = vectors.get_pca_components()
        vectors.normalize()
        vectors.save(config[3], reduced_components)
