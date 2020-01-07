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
    print(line)
    print('-----Should be on a file------')


def gen_header(filepath):
    print_data(filepath, 'train_time,test_raw_time,test_load_time,precision_raw,recall_raw,f1_raw,precision_load,recall_load,f1_load')


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
        cnn.test_raw()
        cnn.test_loading()
        return (
            f"{cnn.train_time}s",
            f"{cnn.test_raw_time}s",
            f"{cnn.test_load_time}s",
            f"{cnn.metrics_raw['class_0']['precision'] * 100}%",
            f"{cnn.metrics_raw['class_0']['recall'] * 100}%",
            f"{cnn.metrics_raw['class_0']['f1'] * 100}%",
            f"{cnn.metrics_raw['class_1']['precision'] * 100}%",
            f"{cnn.metrics_raw['class_1']['recall'] * 100}%",
            f"{cnn.metrics_raw['class_1']['f1'] * 100}%",
            f"{cnn.metrics_raw['class_2']['precision'] * 100}%",
            f"{cnn.metrics_raw['class_2']['recall'] * 100}%",
            f"{cnn.metrics_raw['class_2']['f1'] * 100}%",
            f"{cnn.metrics_load['class_0']['precision'] * 100}%",
            f"{cnn.metrics_load['class_0']['recall'] * 100}%",
            f"{cnn.metrics_load['class_0']['f1'] * 100}%",
            f"{cnn.metrics_load['class_1']['precision'] * 100}%",
            f"{cnn.metrics_load['class_1']['recall'] * 100}%",
            f"{cnn.metrics_load['class_1']['f1'] * 100}%",
            f"{cnn.metrics_load['class_2']['precision'] * 100}%",
            f"{cnn.metrics_load['class_2']['recall'] * 100}%",
            f"{cnn.metrics_load['class_2']['f1'] * 100}%"
        )
    elif network == LSTM:
        pass
    elif network == CNN_LSTM:
        pass


def run(num_runs, filepath, network):
    #precision_values = []
    #recall_values = []
    #accuracy_values = []
    #vectorizing_times = []
    #training_times = []

    for instance in range(num_runs):
        results = run_network(network)
        print('-----------------------------------')
        print_data(filepath, ','.join(results))

        #precision_values.append(results[0])
        #recall_values.append(results[1])
        #accuracy_values.append(results[2])
        #vectorizing_times.append(results[3])
        #training_times.append(results[4])

    #line_results = [
    #    mean(precision_values),
    #    mean(recall_values),
    #    mean(accuracy_values),
    #    mean(vectorizing_times),
    #    mean(training_times)
    #]
    #print_data(filepath, ','.join(line_results))


if __name__ == "__main__":
    gen_header(filepath=filepath)
    for network in networks:
        run(
            num_runs=1,
            filepath=filepath,
            network=network
        )
