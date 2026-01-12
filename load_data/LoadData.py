import tensorflow as tf
import pandas as pd
import numpy as np


class LoadData:

    def __init__(self, cm):
        self.cm = cm

    def data_client(self, cid):

        train = pd.read_pickle(f"{self.cm.path_clients}/{cid}_train.pickle")
        test = pd.read_pickle(f"{self.cm.path_clients}/{cid}_test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']

        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.cm.model_type == "CNN":
            x_train = np.array([x.reshape(self.cm.shape[0], self.cm.shape[1]) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.cm.shape[0], self.cm.shape[1]) for x in x_test.reset_index(drop=True).values])

        return (x_train, y_train), (x_test, y_test), len(x_train)

    def data_server(self):
        train = pd.read_pickle(f"{self.cm.path_server}/train.pickle")
        test = pd.read_pickle(f"{self.cm.path_server}/test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']

        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.cm.model_type == "CNN":
            x_train = np.array([x.reshape(self.cm.shape[0], self.cm.shape[1]) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.cm.shape[0], self.cm.shape[1]) for x in x_test.reset_index(drop=True).values])

        return (x_train, y_train), (x_test, y_test)
