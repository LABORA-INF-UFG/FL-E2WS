import numpy as np
from client.Client import Client
from ml_model.MLModel import Model


class ServerClientList:
    def __init__(self, list_clients, list_distances, cm, optmizer_type):

        self.cm = cm
        self.optmizer_type = optmizer_type
        self.list_clients = list_clients
        self.model = Model.create_model(self.cm)

        self.clients_model_dict = {}
        self.clients_number_data_samples = {}
        self.loss_list = {}

        self.clients_emd = {}
        self.clients_emd_norm = {}
        self.group_map = {}

        self.list_distances = dict(zip(list_clients, list_distances))
        self.init_models()


    @staticmethod
    def min_max_normalize(min_val, max_val, value):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def init_models(self):
        aux_group_map = [5, 5, 3, 4, 2, 1, 3, 5, 5, 4, 3, 3, 3, 5, 1, 4, 1, 2, 2, 1, 5, 4, 5, 3, 4, 5, 4, 3, 2, 1, 4, 1, 3,
                            1, 3, 1, 2, 5, 3, 2, 1, 4, 1, 4, 1, 2, 3, 5, 2, 4, 5, 1, 4, 2, 2, 3, 1, 5, 2, 2, 5, 5, 3, 5, 4, 3,
                            1, 4, 5, 2, 4, 5, 5, 1, 3, 1, 4, 1, 1, 5, 4, 5, 4, 3, 1, 2, 5, 4, 5, 3, 1, 2, 1, 3, 1, 2, 5, 4, 1,
                            1, 3, 5, 1, 2, 3, 4, 4, 2, 2, 4, 3, 3, 2, 3, 2, 5, 3, 2, 2, 2, 1, 4, 2, 3, 3, 1, 2, 3, 2, 5, 4, 1,
                            2, 5, 5, 2, 4, 5, 3, 2, 5, 1, 4, 4, 3, 4, 1, 4, 4, 3]

        for i in self.list_clients:
            tmp_client = Client(i,
                                load_data_constructor=False,
                                cm=self.cm,
                                optmizer_type=self.optmizer_type
                                )
            self.clients_number_data_samples[i] = tmp_client.number_data_samples()
            self.clients_emd[i] = tmp_client.compute_emd()
            self.group_map[i] = aux_group_map[i-1]

            self.loss_list[i] = np.inf
            self.clients_model_dict[i] = None 

        min_emd = min(self.clients_emd.values())
        max_emd = max(self.clients_emd.values())
        for i in self.list_clients:
            emd_value = self.min_max_normalize(
                min_emd,
                max_emd,
                self.clients_emd[i]
            )
            self.clients_emd_norm[i] = emd_value

        print("\nID Clients")
        print(self.list_clients)
        print("\nEMD")
        print(self.clients_emd)
        print("\nlist_distances")
        print(self.list_distances)

    def instantiate(self, selected_clients):
        for i in selected_clients:
            self.clients_model_dict[i] = Client(i,
                                                load_data_constructor=False,
                                                cm=self.cm,
                                                optmizer_type=self.optmizer_type)

    def destroy(self, selected_clients):
        for i in selected_clients:
            self.clients_model_dict[i] = None


    def get_cids(self):
        return list(self.clients_model_dict.keys())

