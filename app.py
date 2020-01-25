import os
from statistics import mean

from deep_learning import ConvolutedNeuralNetwork
from vectorizer import Word2VecModel

# CONSTANTS
CNN = 'cnn'
LSTM = 'lstm'
CNN_LSTM = 'mixed'
WEIGHT_FILEPATH = 'Models/weights.best2.hdf5'

# CONFIGS
num_runs = 10
filepath = 'data.csv'
# networks = [CNN, LSTM, CNN_LSTM]
networks = [CNN]


def print_data(filepath, line):
    with open(filepath, 'a') as file:
        file.write(line)
        file.write('\n')


def gen_header(filepath):
    print_data(filepath, 'train_time,test_time,class_0_precision,class_0_recall,class_0_f1,class_1_precision,class_1_recall,class_1_f1,class_2_precision,class_2_recall,class_2_f1')


def run_network(network_type):
    word2vec = Word2VecModel(
        train_dataset='Data Collection/1_TrainingSet_3Class.csv',
        test_dataset='Data Collection/1_TestSet_3Class.csv',
        class_qtd='3class',
        base_model='Twitter'
    )
    train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()
    word2vec.save(train_vectors, train_labels, test_vectors, test_labels)

    if network_type == CNN:
        cnn = ConvolutedNeuralNetwork(
            train_vectors=train_vectors,
            train_labels=train_labels,
            test_vectors=test_vectors,
            test_labels=test_labels,
            weight_filepath=WEIGHT_FILEPATH
        )
        cnn.train()
        cnn.test()
        return (
            cnn.train_time,
            cnn.test_time,
            cnn.metrics['class_0']['precision'],
            cnn.metrics['class_0']['recall'],
            cnn.metrics['class_0']['f1'],
            cnn.metrics['class_1']['precision'],
            cnn.metrics['class_1']['recall'],
            cnn.metrics['class_1']['f1'],
            cnn.metrics['class_2']['precision'],
            cnn.metrics['class_2']['recall'],
            cnn.metrics['class_2']['f1']
        )
    elif network == LSTM:
        pass
    elif network == CNN_LSTM:
        pass


def run(num_runs, filepath, network):
    train_time_values = []
    test_time_values = []
    class_0_precision_values = []
    class_0_recall_values = []
    class_0_f1_values = []
    class_1_precision_values = []
    class_1_recall_values = []
    class_1_f1_values = []
    class_2_precision_values = []
    class_2_recall_values = []
    class_2_f1_values = []

    for instance in range(num_runs):
        if os.path.exists(WEIGHT_FILEPATH):
            os.remove(WEIGHT_FILEPATH)

        results = run_network(network)

        train_time_values.append(results[0])
        test_time_values.append(results[1])
        class_0_precision_values.append(results[2])
        class_0_recall_values.append(results[3])
        class_0_f1_values.append(results[4])
        class_1_precision_values.append(results[5])
        class_1_recall_values.append(results[6])
        class_1_f1_values.append(results[7])
        class_2_precision_values.append(results[8])
        class_2_recall_values.append(results[9])
        class_2_f1_values.append(results[10])

    line_results = [
        str(mean(train_time_values)),
        str(mean(test_time_values)),
        str(mean(class_0_precision_values)),
        str(mean(class_0_recall_values)),
        str(mean(class_0_f1_values)),
        str(mean(class_1_precision_values)),
        str(mean(class_1_recall_values)),
        str(mean(class_1_f1_values)),
        str(mean(class_2_precision_values)),
        str(mean(class_2_recall_values)),
        str(mean(class_2_f1_values)),
    ]
    print_data(filepath, ','.join(line_results))


if __name__ == "__main__":
    gen_header(filepath=filepath)
    for network in networks:
        run(
            num_runs=1,
            filepath=filepath,
            network=network
        )
