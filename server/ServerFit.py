from server.ServerClientList import ServerClientList


class ServerFit:
    def __init__(self, list_clients, list_distances, cm, optmizer_type):
        self.server_clients = ServerClientList(list_clients=list_clients,
                                               list_distances=list_distances,
                                               cm=cm, optmizer_type=optmizer_type)
        self.selected_clients = []

    def configure_fit(self, selected_clients):
        self.selected_clients = selected_clients

    def fit(self, w_global):
        weight_list = []
        sample_sizes_list = []
        info_list = []
        self.server_clients.instantiate(self.selected_clients)

        for i, cid in enumerate(self.selected_clients):
            print(f"---> [{(i + 1):2}] CID: {cid:3} | S: {self.server_clients.clients_number_data_samples[cid]:5} | D: {self.server_clients.list_distances[cid]:6.2f}")
            weights, size, info = self.server_clients.clients_model_dict[cid].fit(parameters=w_global)

            weight_list.append(weights)
            sample_sizes_list.append(size)
            self.server_clients.loss_list[cid] = info['val_loss']
            info_list.append(info)


        self.server_clients.destroy(self.selected_clients)

        return weight_list, sample_sizes_list, {
            "acc_loss_local": [(cid, info_list[i]) for i, cid in enumerate(self.selected_clients)]}
