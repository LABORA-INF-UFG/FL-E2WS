import sys

import numpy as np


class SinrOpt:

    @staticmethod
    def get_power_index(d, d_min, d_max, ind_power):
        d = np.clip(d, d_min, d_max)
        ratio = (d - d_min) / (d_max - d_min)
        idx_float = ratio * (len(ind_power) - 1)
        idx = int(np.round(idx_float).astype(int))
        idx = np.clip(idx, 0, len(ind_power) - 1)
        return ind_power[idx]

    @staticmethod
    def opt(self):
        print(f"> SINR opt")
        selected_clients = self.selected_clients
        distances = self.comm_model.user_distance.flatten().tolist()

        print(f"selected_clients: {selected_clients}")
        print(f"user_distance: {distances}")

        pos_list = np.arange(len(distances))
        combined_data = list(zip(distances, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        _, _rb_allocation = zip(*sorted_data)
        aux_rb_allocation = [_rb_allocation.index(i) for i in range(len(_rb_allocation))]

        print(f"rb_allocation: {aux_rb_allocation}")

        j_bw = 0
        l_power = len(self.comm_model.user_power) - 1
        m_cpu_freq = len(self.comm_model.cpu_freq) - 1

        _ind_selected_clients = []
        _selected_clients = []
        _rb_allocation = []
        _user_power_allocation = []
        for i, ind in enumerate(selected_clients):
            W = self.comm_model.W[i][j_bw][aux_rb_allocation[i]]           
            if W[l_power][m_cpu_freq] > 0:
                _ind_selected_clients.append(i)
                _selected_clients.append(ind)
                _rb_allocation.append(aux_rb_allocation[i])
                
                ind_power = []
                for i_power in range(len(self.comm_model.user_power)):
                    if W[i_power][m_cpu_freq] > 0:
                        ind_power.append(i_power)

                selected_index = SinrOpt.get_power_index(distances[i], 100, 500, ind_power)
                _user_power_allocation.append(selected_index)
                          

        _user_bandwidth = np.zeros(len(_selected_clients), dtype=int).tolist()
        _user_cpu_freq = np.full(len(_selected_clients), len(self.comm_model.cpu_freq) - 1).tolist()

        print(f"_user_power_allocation: {_user_power_allocation}")
        print(f"_selected_clients: {_selected_clients}")
        for i, ind in enumerate(_ind_selected_clients):
            print(f"[{(i+1):2}] Device {_selected_clients[i]:3} - "
                  f"Channel: {_rb_allocation[i]:2} - "        
                  f"distance: {float(distances[ind]):6.2f} - W: {self.comm_model.W[ind, _user_bandwidth[i], _rb_allocation[i], _user_power_allocation[i], _user_cpu_freq[i]]:6.6f}")
        
        return _ind_selected_clients, _selected_clients, _user_bandwidth, _rb_allocation, _user_power_allocation, _user_cpu_freq


