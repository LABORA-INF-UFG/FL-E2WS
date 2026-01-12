import numpy as np
import pandas as pd
import sys
from load_data.LoadData import LoadData
from ml_model.MLModel import Model
from client.EmdCompute import EMD_Compute
import tensorflow as tf


class Client:
    def __init__(self, cid, load_data_constructor, cm, optmizer_type):
        self.cid = int(cid)
        self.load_data_constructor = load_data_constructor
        self.cm = cm
        self.optmizer_type = optmizer_type

        self.model = Model.create_model(cm=self.cm)
        self.load_data = LoadData(cm=self.cm)

        self.emd_cmp = EMD_Compute()

        if self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = (None, None), (None, None), 0

    def number_data_samples(self):
        (self.x_train, _), (_, _), _ = self.load_data.data_client(self.cid)
        return len(self.x_train)

    def compute_emd(self):
        (_, y_train), (_, _), _ = self.load_data.data_client(self.cid)
        value_emd = self.emd_cmp.compute_value(y_train.values)    
        return value_emd

    def fit(self, parameters, config=None):
        if self.optmizer_type != 'FedProx':
            return self.fit_default(parameters, config)
        else:     
            return self.fit_fed_prox(parameters, config)

    def fit_default(self, parameters, config=None):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)

        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 epochs=1,
                                 batch_size=128,
                                 validation_data=(self.x_test, self.y_test),
                                 verbose=False)
        sample_size = len(self.x_train)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
                                                       "val_loss": history.history['val_loss'][-1]}
    def fit_fed_prox(self, parameters, config=None):
        _mu = 0.1

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)

        self.model.set_weights(parameters)

        batch_size = 128
        num_batches = len(self.x_train) // batch_size

        for epoch in range(1):  # Fixed number of epochs
            batch_count = 0
            for i in range(num_batches):
                x_batch = self.x_train[i * batch_size:(i + 1) * batch_size]
                y_batch = self.y_train[i * batch_size:(i + 1) * batch_size]

                with tf.GradientTape() as tape:

                    if self.cm.model_type == "MLP":
                        x_batch = tf.reshape(x_batch, (-1, 784))


                    predictions = self.model(x_batch, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)

                    # FedProx term: L2 distance between local and global weights
                    prox_term = sum(
                        tf.reduce_sum(tf.square(w1 - w2)) for w1, w2 in zip(self.model.trainable_weights, parameters))
                    loss = loss + (_mu / 2) * prox_term

                grads = tape.gradient(loss, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                batch_count += 1
                if batch_count >= num_batches:
                    break

            sample_size = len(self.x_train)
            loss, accuracy = self.evaluate(self.model.get_weights())

            if not self.load_data_constructor:
                (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

            return self.model.get_weights(), sample_size, {"val_accuracy": accuracy, "val_loss": loss}

    def evaluate(self, parameters):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = self.load_data.data_client(self.cid)

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.number_samples = (None, None), (None, None), 0

        return loss, accuracy
