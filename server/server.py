import numpy as np
import pandas as pd
from ml_model.ml_model import Model
from client.client import Client


class Server:
    def __init__(self, n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                 path_server, path_clients, shape, model_type, parallel_processing=False):

        self.n_rounds = n_rounds
        self.total_number_clients = total_number_clients
        self.min_fit_clients = min_fit_clients
        self.load_client_data_constructor = load_client_data_constructor
        self.parallel_processing = parallel_processing

        self.path_server = path_server
        self.path_clients = path_clients
        self.shape = shape
        self.model_type = model_type

        self.server_round = 0
        self.selected_clients = []

        if self.model_type == "MLP":
            self.model = Model.create_model_mlp()
        else:
            self.model = Model.create_model_cnn(self.shape)

        self.w_global = self.model.get_weights()

        self.clients_model_list = []
        self.clients_number_data_samples = []
        self.clients_acc = []
        self.clients_loss = []

        self.count_of_client_selected = []
        self.count_of_client_uploads = []

        self.evaluate_list = {"distributed": {"loss": [], "accuracy": []}, "centralized": {"loss": [], "accuracy": []}}

        self.create_models()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

    def create_models(self):
        for i in range(self.total_number_clients):
            self.clients_model_list.append(Client(i + 1, self.load_client_data_constructor, self.path_clients, self.shape, self.model_type))
            self.clients_number_data_samples.append(self.clients_model_list[i].number_data_samples())
            self.clients_acc.append(0)
            self.clients_loss.append(np.inf)
            self.count_of_client_selected.append(0)
            self.count_of_client_uploads.append(0)

    def load_data(self):
        train = pd.read_pickle(f"{self.path_server}/train.pickle")
        test = pd.read_pickle(f"{self.path_server}/test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']
        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.model_type == "CNN":
            x_train = np.array([x.reshape(self.shape) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.shape) for x in x_test.reset_index(drop=True).values])

        return (x_train, y_train), (x_test, y_test)

    def aggregate_fit(self, parameters, sample_sizes):
        self.w_global = []
        for weights in zip(*parameters):
            weighted_sum = 0
            total_samples = sum(sample_sizes)
            for i in range(len(weights)):
                weighted_sum += weights[i] * sample_sizes[i]
            self.w_global.append(weighted_sum / total_samples)

    def configure_fit(self):
        self.selected_clients = np.random.permutation(list(range(self.total_number_clients)))[:self.min_fit_clients]

    def fit(self):

        weight_list = []
        sample_sizes_list = []
        info_list = []

        if self.parallel_processing:
            pass
        else:

            for i, pos in enumerate(self.selected_clients):
                print(f"-------> [{i + 1}] (R: {self.server_round + 1}/{self.n_rounds}) CID: {pos}")
                weights, size, info = self.clients_model_list[pos].fit(parameters=self.w_global)
                weight_list.append(weights)
                sample_sizes_list.append(size)
                info_list.append(info)
                self.clients_acc[pos] = info['val_accuracy']
                self.clients_loss[pos] = info['val_loss']

        return weight_list, sample_sizes_list, {
            "acc_loss_local": [(pos + 1, info_list[i]) for i, pos in enumerate(self.selected_clients)]}

    def distributed_evaluation(self):

        loss_list = []
        accuracy_list = []

        if self.parallel_processing:
            pass
        else:
            for i in range(self.total_number_clients):
                loss, accuracy = self.clients_model_list[i].evaluate(parameters=self.w_global)
                # print(f"Evaluate - CID: {i+1} - accuracy: {accuracy}")
                loss_list.append(loss)
                accuracy_list.append(accuracy)

        loss = sum(loss_list) / len(loss_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)
        self.evaluate_list["distributed"]["loss"].append(loss)
        self.evaluate_list["distributed"]["accuracy"].append(accuracy)

        return loss, accuracy, {"accuracy_list": [(i + 1, accuracy) for i, accuracy in enumerate(accuracy_list)]}

    def centralized_evaluation(self):
        self.model.set_weights(self.w_global)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)
        self.evaluate_list["centralized"]["loss"].append(loss)
        self.evaluate_list["centralized"]["accuracy"].append(accuracy)
        return loss, accuracy
