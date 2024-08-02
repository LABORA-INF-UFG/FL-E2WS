import numpy as np
import pandas as pd
from ml_model.ml_model import Model


class Client:
    def __init__(self, cid, load_data_constructor, path, shape, model_type):
        self.cid = int(cid)
        self.load_data_constructor = load_data_constructor

        self.path = path
        self.shape = shape
        self.model_type = model_type

        if self.model_type == "MLP":
            self.model = Model.create_model_mlp()
        else:
            self.model = Model.create_model_cnn(self.shape)

        if self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

    def load_data(self):

        train = pd.read_pickle(f"{self.path}/{self.cid}_train.pickle")
        test = pd.read_pickle(f"{self.path}/{self.cid}_test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']

        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.model_type == "CNN":
            x_train = np.array([x.reshape(self.shape) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.shape) for x in x_test.reset_index(drop=True).values])

        return (x_train, y_train), (x_test, y_test)

    def number_data_samples(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        return len(self.x_train)

    def fit(self, parameters, config=None):
        # print(f"CID: {self.cid}")

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=128,
                                 validation_data=(self.x_test, self.y_test), verbose=False)
        sample_size = len(self.x_train)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
                                                       "val_loss": history.history['val_loss'][-1]}

    def evaluate(self, parameters):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return loss, accuracy
