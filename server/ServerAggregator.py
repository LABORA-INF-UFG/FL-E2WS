import sys
import numpy as np
from load_data.LoadData import LoadData
from ml_model.MLModel import Model


class ServerAgregador:
    def __init__(self, csf, parameters=None):

        self.csf = csf
        self.load_data = LoadData(self.csf.cm)
        self.clients_emd = {}
        self.clients_emd_norm = {}

        self.model = Model.create_model(self.csf.cm)
        self.total_params = self.model.count_params()

        if parameters is None:
            self.w_global = self.model.get_weights()
        else:
            self.w_global = parameters

        (_, _), (self.x_test, self.y_test) = self.load_data.data_server()
        self.evaluate_list = {"centralized": {"loss": [], "accuracy": []}}

    def set_clients_emd(self, clients_emd, clients_emd_norm):
        self.clients_emd = clients_emd
        self.clients_emd_norm = clients_emd_norm

    def aggregate_fit(self, selected_clients, parameters, sample_sizes):        
        if self.csf.optmizer_type == "E2WS":
            print("agg-EMD")
            emd_list = np.array([self.clients_emd[k] for k in selected_clients])
            sample_sizes = sample_sizes * (1 / emd_list)
            self.aggregate_fit_default(parameters, sample_sizes)
        else:
            print("agg-default")
            self.aggregate_fit_default(parameters, sample_sizes)

    def aggregate_fit_default(self, parameters, sample_sizes):
        self.w_global = []
        for weights in zip(*parameters):
            weighted_sum = 0
            total_samples = sum(sample_sizes)
            for i in range(len(weights)):
                weighted_sum += weights[i] * sample_sizes[i]
            self.w_global.append(weighted_sum / total_samples)

    def centralized_evaluation(self):
        self.model.set_weights(self.w_global)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        self.evaluate_list["centralized"]["loss"].append(loss)
        self.evaluate_list["centralized"]["accuracy"].append(accuracy)
        return loss, accuracy

    def print_evaluate(self, loss=False):
        print(f"accuracy: \n{self.evaluate_list['centralized']['accuracy']}")
        if loss:
            print(f"loss: \n{self.evaluate_list['centralized']['loss']}")
