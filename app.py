import os

from deep_learning import (
    ConvolutedNeuralNetwork,
    LongShortTermMemoryNetwork,
    CombinedNeuralNetworks
)
from vectorizer import Word2VecModel

# CONSTANTS
CNN = 'cnn'
LSTM = 'lstm'
CNN_LSTM = 'mixed'
W2V = 'Pt-BR_Word2Vec'
GLOVE = 'Pt-BR_GloVe'
FAST = 'Pt-BR_FastText'
CLASSES = '2class'
WEIGHT_FILEPATH = 'Models/weights.best2.hdf5'
HEADER = '''\
vectorizer,\
dimensions,\
classifier,\
vector_time (s),\
train_time (s),\
accuracy,\
precision,\
recall,\
f1\
'''

# CONFIGS
num_runs = 10
filepath = 'data.csv'
configs = [
    # (W2V, CNN, 50),
    # (W2V, CNN, 100),
    # (W2V, CNN, 300),
    (W2V, CNN, 600),
    (W2V, CNN, 1000),
    (W2V, LSTM, 50),
    (W2V, LSTM, 100),
    (W2V, LSTM, 300),
    # (W2V, LSTM, 600),
    # (W2V, LSTM, 1000),
    # (W2V, CNN_LSTM, 50),
    # (W2V, CNN_LSTM, 100),
    # (W2V, CNN_LSTM, 300),
    (W2V, CNN_LSTM, 600),
    (W2V, CNN_LSTM, 1000),
    (GLOVE, CNN, 50),
    (GLOVE, CNN, 100),
    # (GLOVE, CNN, 300),
    # (GLOVE, CNN, 600),
    # (GLOVE, CNN, 1000),
    # (GLOVE, LSTM, 50),
    # (GLOVE, LSTM, 100),
    # (GLOVE, LSTM, 300),
    # (GLOVE, LSTM, 600),
    # (GLOVE, LSTM, 1000),
    # (GLOVE, CNN_LSTM, 50),
    # (GLOVE, CNN_LSTM, 100),
    # (GLOVE, CNN_LSTM, 300),
    # (GLOVE, CNN_LSTM, 600),
    # (GLOVE, CNN_LSTM, 1000),
    # (FAST, CNN, 50),
    # (FAST, CNN, 100),
    # (FAST, CNN, 300),
    # (FAST, CNN, 600),
    # (FAST, CNN, 1000),
    # (FAST, LSTM, 50),
    # (FAST, LSTM, 100),
    # (FAST, LSTM, 300),
    # (FAST, LSTM, 600),
    # (FAST, LSTM, 1000),
    # (FAST, CNN_LSTM, 50),
    # (FAST, CNN_LSTM, 100),
    # (FAST, CNN_LSTM, 300),
    # (FAST, CNN_LSTM, 600),
    # (FAST, CNN_LSTM, 1000),
]


def print_data(filepath, line):
    with open(filepath, 'a') as file:
        file.write(line)
        file.write('\n')


def gen_header(filepath):
    print_data(filepath, HEADER)


def run_network(config):
    vectorizer = config[0]
    network_type = config[1]
    dimensions = config[2]

    vector = Word2VecModel(
        train_dataset='Data Collection/2class_training_br.csv',
        test_dataset='Data Collection/2class_testing_br.csv',
        class_qtd='2class',
        base_model=vectorizer,
        set_dimensions=dimensions
    )

    vectorized = vector.vectorize()
    vector.save(
        vectorized['train_vectors'],
        vectorized['train_labels'],
        vectorized['test_vectors'],
        vectorized['test_labels']
    )

    if network_type == CNN:
        network = ConvolutedNeuralNetwork(
            train_vectors=vectorized['train_vectors'],
            train_labels=vectorized['train_labels'],
            test_vectors=vectorized['test_vectors'],
            test_labels=vectorized['test_labels'],
            weight_filepath=WEIGHT_FILEPATH,
            vectorizing_time=vectorized['vectorizing_time']
        )
    elif network_type == LSTM:
        network = LongShortTermMemoryNetwork(
            train_vectors=vectorized['train_vectors'],
            train_labels=vectorized['train_labels'],
            test_vectors=vectorized['test_vectors'],
            test_labels=vectorized['test_labels'],
            weight_filepath=WEIGHT_FILEPATH,
            vectorizing_time=vectorized['vectorizing_time']
        )
    elif network_type == CNN_LSTM:
        network = CombinedNeuralNetworks(
            train_vectors=vectorized['train_vectors'],
            train_labels=vectorized['train_labels'],
            test_vectors=vectorized['test_vectors'],
            test_labels=vectorized['test_labels'],
            weight_filepath=WEIGHT_FILEPATH,
            vectorizing_time=vectorized['vectorizing_time']
        )
    network.train()
    network.test()
    return (
        network.vectorizing_time,
        network.train_time,
        network.metrics['accuracy'],
        network.metrics['precision'],
        network.metrics['recall'],
        network.metrics['f1'],
    )


def run(num_runs, filepath, config):
    for instance in range(num_runs):
        print(f'Rodada {instance} de {num_runs}')
        if os.path.exists(WEIGHT_FILEPATH):
            os.remove(WEIGHT_FILEPATH)

        results = run_network(config)

        line_results = [
            str(config[0]),
            str(config[2]),
            str(config[1]),
            str(results[0]),
            str(results[1]),
            str(results[2]),
            str(results[3]),
            str(results[4]),
            str(results[5]),
        ]
        print_data(filepath, ','.join(line_results))


if __name__ == "__main__":
    gen_header(filepath=filepath)
    for config in configs:
        run(
            num_runs=10,
            filepath=filepath,
            config=config
        )
