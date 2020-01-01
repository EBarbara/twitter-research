from statistics import mean


import deep_learning
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
networks = [CNN, LSTM, CNN_LSTM]


def print_data(filepath, line):
    pass


def gen_header(filepath):
    print_data(filepath, 'precision, recall, accuracy, vectorizing, training')


def run_network(network):
    word2vec = Word2VecModel(
        train_dataset='Data Collection/1_TrainingSet_3Class.csv',
        test_dataset='Data Collection/1_TestSet_3Class.csv',
        class_qtd='3class',
        base_model='Twitter'
    )
    train_vectors, train_labels, test_vectors, test_labels = word2vec.vectorize()
    word2vec.save(train_vectors, train_labels, test_vectors, test_labels)

    if network == CNN:
        cnn = ConvolutedNeuralNetwork(
            train_vectors=train_vectors,
            train_labels=train_labels
        )
        deep_learning.train_network(
            network=cnn,
            test_vectors=test_vectors,
            test_labels=test_labels,
            weight_filepath=WEIGHT_FILEPATH,
            hist_filepath='Models/Accuracy-History-CNN-2class.pickle'
        )
        deep_learning.test_network(
            network=cnn,
            weight_filepath=WEIGHT_FILEPATH,
            loss_function='categorical_crossentropy',
            optimizer_function='adam',
            metrics=['accuracy'],
            test_vectors=test_vectors,
            test_labels=test_labels
        )
    elif network == LSTM:
        pass
    elif network == CNN_LSTM:
        pass


def run(num_runs, filepath, network):
    precision_values = []
    recall_values = []
    accuracy_values = []
    vectorizing_times = []
    training_times = []

    for instance in range(num_runs):
        results = run_network(network)

        precision_values.append(results[0])
        recall_values.append(results[1])
        accuracy_values.append(results[2])
        vectorizing_times.append(results[3])
        training_times.append(results[4])

    line_results = [
        mean(precision_values),
        mean(recall_values),
        mean(accuracy_values),
        mean(vectorizing_times),
        mean(training_times)
    ]
    print_data(filepath, ','.join(line_results))


if __name__ == "__main__":
    gen_header(filepath=filepath)
    for network in networks:
        run(
            num_runs=num_runs,
            filepath=filepath,
            network=network
        )
