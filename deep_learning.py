import pickle
import time

import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def train_network(
    network,
    test_vectors,
    test_labels,
    weight_filepath,
    hist_filepath=None
):
    model = network.model

    test_labels = keras.utils.to_categorical(
        test_labels,
        num_classes=network.qtd_classes
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
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
        validation_data=(test_vectors, test_labels),  # faz sentido... ele precisaria do gabarito para escolher o melhor modelo
        callbacks=callbacks_list
    )

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
        metrics=metrics
    )
    # Computer confusion matrix, precision, recall.
    pred = model.predict(test_vectors, batch_size=64) 
    pred_labels = np.argmax(pred, axis=1)
    real_labels = [int(item) for item in test_labels]

    accuracy = len(np.where(pred_labels == np.array(real_labels))[0])/len(real_labels) * 100
    print(f'Test Accuracy %: {accuracy}\n')

    print('Confusion matrix:')
    print(confusion_matrix(real_labels, pred_labels))

    print('\n')
    print(classification_report(real_labels, pred_labels, digits=3))


class ConvolutedNeuralNetwork():
    def __init__(self, train_vectors, train_labels):
        self.start_time = time.clock()

        self.qtd_classes = len(list(set((train_labels))))
        self.train_vectors = train_vectors
        self.train_labels = keras.utils.to_categorical(
            train_labels,
            num_classes=self.qtd_classes
        )

        self.vector_length = len(train_vectors[0, :, 0])
        self.vector_dimension = len(train_vectors[0, 0, :])

        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Reshape(
                (self.vector_length, self.vector_dimension, 1),
                input_shape=(self.vector_length, self.vector_dimension)
            )
        )
        self.model.add(
            keras.layers.Conv2D(
                100,
                (2, self.vector_dimension),
                strides=(1, 1),
                padding='valid',
                activation='relu',
                use_bias=True
            )
        )
        output = self.model.output_shape
        self.model.add(
            keras.layers.MaxPooling2D(pool_size=(output[1], output[2]))
        )
        self.model.add(keras.layers.Dropout(.5))
        self.model.add(keras.layers.Flatten())
        self.model.add(
            keras.layers.Dense(self.qtd_classes, activation='softmax')
        )
        optimizer = keras.optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0
        )
        # optimizer = keras.optimizers.SGD()
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


class LongShortTermMemoryNetwork():
    def __init__(self):
        pass
