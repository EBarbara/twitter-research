from collections import Counter
import pickle
import time

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train_network(
        network,
        test_vectors,
        test_labels,
        weight_filepath,
        hist_filepath=None
):
    model = network.model

    test_labels = to_categorical(
        test_labels,
        num_classes=network.qtd_classes
    )

    checkpoint = ModelCheckpoint(
        weight_filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    callbacks_list = [checkpoint]
    hist = model.fit(
        network.train_vectors,
        network.train_labels,
        epochs=20,
        batch_size=64,
        shuffle=False,
        validation_data=(test_vectors, test_labels),
        callbacks=callbacks_list
    )

    # Plot training & validation accuracy values
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Print the maximum acc_val and its corresponding epoch
    index = np.argmax(hist.history['val_acc'])
    print(f"The optimal epoch size: {hist.epoch[index]}, The value of high accuracy {np.max(hist.history['val_acc'])}")
    print('\n')
    print(f"Computation Time {time.clock() - network.start_time} seconds")
    print('\n')

    # Save the history accuracy results.
    if hist_filepath:
        with open(hist_filepath, 'wb') as f:
            pickle.dump([hist.epoch, hist.history['acc'], hist.history['val_acc']], f)


def test_network(
        network,
        weight_filepath,
        loss_function,
        optimizer_function,
        metrics,
        test_vectors,
        test_labels
):
    model = network.model
    # load weights, predict based on the best model, compute accuracy, precision, recall, f-score, confusion matrix.
    model.load_weights(weight_filepath)
    # Compile model (required to make predictions)
    model.compile(
        loss=loss_function,
        optimizer=optimizer_function,
        metrics=metrics,
    )
    # Computer confusion matrix, precision, recall.
    pred = model.predict(test_vectors, batch_size=64)
    pred_labels = np.argmax(pred, axis=1)
    real_labels = [int(item) for item in test_labels]

    accuracy = len(np.where(pred_labels == np.array(real_labels))[0]) / len(real_labels) * 100
    print(f'Test Accuracy %: {accuracy}\n')

    print('Confusion matrix:')
    print(confusion_matrix(real_labels, pred_labels))

    print('\n')
    print(classification_report(real_labels, pred_labels, digits=3))


class ConvolutedNeuralNetwork:
    def __init__(
            self,
            train_vectors,
            train_labels,
            test_vectors,
            test_labels,
            weight_filepath,
            optimizer_alg='ADAM'
    ):
        start_time = time.clock()

        self.loss_function = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.train_time = 0
        self.metrics_raw = None
        self.confusion_raw = None
        self.test_raw_time = 0
        self.metrics_load = None
        self.confusion_load = None
        self.test_load_time = 0

        self.weight_filepath = weight_filepath

        self.qtd_classes = len(list(set(train_labels)))
        self.train_vectors = train_vectors
        self.train_labels = to_categorical(
            train_labels,
            num_classes=self.qtd_classes
        )
        self.test_vectors = test_vectors
        self.test_labels = to_categorical(
            test_labels,
            num_classes=self.qtd_classes
        )
        self.real_labels = np.array([int(item) for item in test_labels])

        self.vector_length = len(train_vectors[0, :, 0])
        self.vector_dimension = len(train_vectors[0, 0, :])

        self.model = Sequential()
        self.model.add(
            Reshape(
                (self.vector_length, self.vector_dimension, 1),
                input_shape=(self.vector_length, self.vector_dimension)
            )
        )
        self.model.add(
            Conv2D(
                200,
                (2, self.vector_dimension),
                strides=(1, 1),
                padding='valid',
                activation='relu',
                use_bias=True
            )
        )
        output = self.model.output_shape
        self.model.add(MaxPooling2D(pool_size=(output[1], output[2])))
        self.model.add(Dropout(.5))
        self.model.add(Flatten())
        self.model.add(Dense(self.qtd_classes, activation='softmax'))
        if optimizer_alg == 'SGD':
            self.optimizer = SGD()
        else:  # default optimizer is ADAM
            self.optimizer = Adam(
                lr=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                decay=0.0
            )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        self.load_time = time.clock() - start_time

    def train(self):
        start_time = time.clock()
        checkpoint = ModelCheckpoint(
            self.weight_filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        callbacks_list = [checkpoint]
        self.model.fit(
            self.train_vectors,
            self.train_labels,
            epochs=20,
            batch_size=64,
            shuffle=False,
            validation_data=(self.test_vectors, self.test_labels),
            callbacks=callbacks_list
        )

        self.train_time = (time.clock() - start_time)

    def test_raw(self):
        start_time = time.clock()

        pred = self.model.predict(self.test_vectors, batch_size=64)
        pred_labels = np.argmax(pred, axis=1)

        self.metrics_raw = self.calculate_metrics(pred_labels)
        self.confusion_raw = confusion_matrix(self.real_labels, pred_labels)
        self.test_raw_time = (time.clock() - start_time)

    def test_loading(self):
        start_time = time.clock()
        self.model.load_weights(self.weight_filepath)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics
        )

        pred = self.model.predict(self.test_vectors, batch_size=64)
        pred_labels = np.argmax(pred, axis=1)

        self.metrics_load = self.calculate_metrics(pred_labels)
        self.confusion_load = confusion_matrix(self.real_labels, pred_labels)
        self.test_load_time = (time.clock() - start_time)

    def calculate_metrics(self, pred_labels):
        result = {}
        if self.qtd_classes == 2:
            result = self.evaluate_class(pred_labels, 1)
            result['type'] = 'binary'
        else:
            result['type'] = 'multiclass'
            for i in range(self.qtd_classes):
                result[f'class_{i}'] = self.evaluate_class(pred_labels, i)
        return result

    def evaluate_class(self, pred_labels, class_id):
        result_prediction = []
        for pred, real in np.nditer([pred_labels, self.real_labels]):
            if pred == class_id and real == class_id:
                result_prediction.append('True positive')
            elif pred == class_id:
                result_prediction.append('False positive')
            elif real == class_id:
                result_prediction.append('False negative')
            else:
                result_prediction.append('True negative')
        result_count = Counter(result_prediction)
        true_positives = result_count['True positive']
        positives = true_positives + result_count['False positive']
        relevants = true_positives + result_count['False negative']

        precision = true_positives / positives
        recall = true_positives / relevants
        f1 = 2*((precision * recall)/(precision + recall))

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class LongShortTermMemoryNetwork:
    def __init__(self):
        pass
