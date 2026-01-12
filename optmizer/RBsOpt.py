import sys
import numpy as np
import pulp as pl
from pulp import PULP_CBC_CMD
import re


class RBsOpt:

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
        selected_clients = self.selected_clients
        metric_clients = list(self.metric_clients.values())
        print(f"selected_clients: {selected_clients}")        

        print(f"> MILP")
        
        # Creation of the assignment problem
        model = pl.LpProblem("Max_Prob", pl.LpMaximize)

        # Decision Variables
        x = [[pl.LpVariable(f"x_{i}_{k}", cat=pl.LpBinary) for k in
              range(self.csf.rb_number)] for i in range(len(selected_clients))]

        j_bw = 0
        l_power = len(self.comm_model.user_power) - 1
        m_cpu_freq = len(self.comm_model.cpu_freq) - 1
        
        model += pl.lpSum(
            (self.comm_model.W[i][j_bw][k][l_power][m_cpu_freq] * x[i][k])
            for k in range(self.csf.rb_number)
            for i in range(len(selected_clients))), "Max"

        model += pl.lpSum([x[i][k] for k in range(self.csf.rb_number) for i in
                           range(len(selected_clients))]) <= self.csf.min_fit_clients, f"min_fit_clients"

        # Constraints: Each customer is assigned to exactly one channel and power
        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][k]for k in
                              range(self.csf.rb_number)) >= 0, f"Customer_Channel_Constraints_{i} >= 0"

        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][k] for k in
                              range(self.csf.rb_number)) <= 1, f"Customer_Channel_Constraints_{i} <= 1"

        # Constraints: Each channel is assigned to exactly one customer
        for k in range(self.csf.rb_number):
            model += pl.lpSum(x[i][k] for i in
                              range(len(selected_clients))) >= 0, f"Channel_Customer_Constraints_{k} >= 0"

        for k in range(self.csf.rb_number):
            model += pl.lpSum(x[i][k] for i in
                              range(len(selected_clients))) <= 1, f"Channel_Customer_Constraints_{k} <= 1"

        for i in range(len(selected_clients)):
            for k in range(self.csf.rb_number):
                model += x[i][k] * self.comm_model.q[i][j_bw][k][l_power] <= self.csf.error_rate_requirement, f"Delay_Constraints_{i}_{k}"
                model += x[i][k] * self.comm_model.user_delay_upload[i][j_bw][k][l_power] <= self.csf.delay_requirement, f"Packet_Error_Rate_Constraints_{i}_{k}"

        ################
        # Solving the problem
        status = model.solve(pl.PULP_CBC_CMD(msg=0))   

        _ind_selected_clients = []
        _selected_clients = []
        _rb_allocation = []
        _user_power_allocation = []
        for var in model.variables():
            if pl.value(var) == 1:               
                indices = [int(i) for i in re.findall(r'\d+', var.name)]
               
                _ind_selected_clients.append(indices[0])
                _selected_clients.append(selected_clients[indices[0]])
                _rb_allocation.append(indices[1])

                ######
                W = self.comm_model.W[indices[0]][j_bw][indices[1]]
                ind_power = []
                for i_power in range(len(self.comm_model.user_power)):
                    if W[i_power][m_cpu_freq] > 0:
                        ind_power.append(i_power)

                d_ue = self.comm_model.user_distance[indices[0]]
                selected_index = RBsOpt.get_power_index(d_ue, 100, 500, ind_power)
                _user_power_allocation.append(selected_index)
                ######

        _user_bandwidth = np.zeros(len(_selected_clients), dtype=int).tolist()
        _user_cpu_freq = np.full(len(_selected_clients), len(self.comm_model.cpu_freq) - 1).tolist()

        for i, ind in enumerate(_ind_selected_clients):
            print(f"[{(i+1):2}] Device {_selected_clients[i]:3} - "
                  f"Channel: {_rb_allocation[i]:2} - "        
                  f"distance: {float(self.comm_model.user_distance[ind]):6.2f} - W: {self.comm_model.W[ind, _user_bandwidth[i], _rb_allocation[i], _user_power_allocation[i], _user_cpu_freq[i]]:6.6f}")

        print(f"_ind_selected_clients: {_ind_selected_clients}")
        print(f"_selected_clients: {_selected_clients}")
        print(f"_user_bandwidth: {_user_bandwidth}")
        print(f"_rb_allocation: {_rb_allocation}")
        print(f"_user_power_allocation: {_user_power_allocation}")
        print(f"_user_cpu_freq: {_user_cpu_freq}")
 
        return _ind_selected_clients, _selected_clients, _user_bandwidth, _rb_allocation, _user_power_allocation, _user_cpu_freq
