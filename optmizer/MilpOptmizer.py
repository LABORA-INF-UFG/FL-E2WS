import sys
import numpy as np
import pulp as pl
from pulp import PULP_CBC_CMD
import re


class MilpOpt:

    @staticmethod
    def opt(self):
        selected_clients = self.selected_clients
        metric_clients = list(self.metric_clients.values())
        print(f"selected_clients: {selected_clients}")        

        print(f"> MILP")
       
        # Creation of the assignment problem
        model = pl.LpProblem("Max_Prob", pl.LpMaximize)

        # Decision Variables
        x = [[[[[pl.LpVariable(f"x_{i}_{j}_{k}_{l}_{m}", cat=pl.LpBinary) for m in range(len(self.comm_model.cpu_freq))] for l in
                range(len(self.comm_model.user_power))] for k in range(self.csf.rb_number)] for j in range(len(self.comm_model.user_bandwidth))] for i in
             range(len(selected_clients))]

        # Objective function
        lmbda = 0 if self.csf.fixed_parameters is True else self.csf.lmbda * 100
        beta = 0 if self.csf.fixed_parameters is True else self.csf.beta * 100
        gama = 0 if self.csf.fixed_parameters is True else self.csf.gama * 100


        #sys.exit()
        model += pl.lpSum(
            (self.comm_model.W[i][j][k][l][m] * x[i][j][k][l][m])
            - (lmbda * self.comm_model.total_energy[i][j][k][l][m] * x[i][j][k][l][m])
            + beta * (1.0 - metric_clients[i]) * x[i][j][k][l][m]
            + gama * x[i][j][k][l][m]
            for m in range(len(self.comm_model.cpu_freq))
            for l in range(len(self.comm_model.user_power))
            for k in range(self.csf.rb_number)
            for j in range(len(self.comm_model.user_bandwidth))
            for i in range(len(selected_clients))), "Max"

        model += pl.lpSum(
            x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for k in range(self.csf.rb_number)
            for j in range(len(self.comm_model.user_bandwidth)) for i in range(len(selected_clients))) <= self.csf.min_fit_clients, f"min_fit_clients"

        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for k in
                              range(self.csf.rb_number) for j in
                              range(len(self.comm_model.user_bandwidth))) >= 0, f"Customer_Channel_Constraints_{i} >= 0"

        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for k in
                              range(self.csf.rb_number) for j in
                              range(len(self.comm_model.user_bandwidth))) <= 1, f"Customer_Channel_Constraints_{i} <= 1"

        for k in range(self.csf.rb_number):
            model += pl.lpSum(x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for i in
                              range(len(selected_clients)) for j in
                              range(len(self.comm_model.user_bandwidth))) >= 0, f"Channel_Customer_Constraints_{k} >= 0"

        for k in range(self.csf.rb_number):
            model += pl.lpSum(x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for i in
                              range(len(selected_clients)) for j in
                              range(len(self.comm_model.user_bandwidth))) <= 1, f"Channel_Customer_Constraints_{k} <= 1"

        for i in range(len(selected_clients)):
            for j in range(len(self.comm_model.user_bandwidth)):
                for k in range(self.csf.rb_number):
                    for l in range(len(self.comm_model.user_power)):
                        for m in range(len(self.comm_model.cpu_freq)):
                            model += x[i][j][k][l][m] * self.comm_model.q[i][j][k][l] <= self.csf.error_rate_requirement, f"Packet_Error_Rate_Constraints_{i}_{j}_{k}_{l}_{m}"

                            model += x[i][j][k][l][m] * self.comm_model.user_delay_upload[i][j][k][l] <= self.csf.delay_requirement, f"Delay_Constraints_{i}_{j}_{k}_{l}_{m}"
                            """
                            model += x[i][j][k][l][m] * self.comm_model.total_energy[i][j][k][l][
                                m] <= self.csf.energy_requirement, f"Energy_Constraints_{i}_{j}_{k}_{l}_{m}"
                            """

        if self.csf.fixed_parameters is False:
            max_bandwidth = self.csf.rb_number * 1
            model += pl.lpSum((np.tile(np.repeat(self.comm_model.user_bandwidth, (len(self.comm_model.user_power) * self.csf.rb_number * len(self.comm_model.cpu_freq))),
                                       len(selected_clients)).reshape(
                (len(selected_clients), len(self.comm_model.user_bandwidth), self.csf.rb_number, len(self.comm_model.user_power), len(self.comm_model.cpu_freq)))[i][j][k][l][m]) *
                              x[i][j][k][l][m] for m in range(len(self.comm_model.cpu_freq)) for l in range(len(self.comm_model.user_power)) for k in
                              range(self.csf.rb_number) for j in range(len(self.comm_model.user_bandwidth)) for i in
                              range(len(selected_clients))) <= max_bandwidth, f"Bandwidth Budget"

        ################
        # Solving the problem
        status = model.solve(pl.PULP_CBC_CMD(msg=0))

        _ind_selected_clients = []
        _selected_clients = []
        _user_bandwidth = []
        _rb_allocation = []
        _user_power_allocation = []
        _user_cpu_freq = []
        for var in model.variables():
            if pl.value(var) == 1:
                indices = [int(i) for i in re.findall(r'\d+', var.name)]
                # print(indices)
                _ind_selected_clients.append(indices[0])
                _selected_clients.append(selected_clients[indices[0]])
                _user_bandwidth.append(indices[1])
                _rb_allocation.append(indices[2])
                _user_power_allocation.append(indices[3])
                _user_cpu_freq.append(indices[4])
   
        print("-------------------------")
        for i, ind in enumerate(_ind_selected_clients):
            print(f"[{(i+1):2}] Device {_selected_clients[i]:3} - "
                  f"Channel: {_rb_allocation[i]:2} - "
                  f"power: {float(self.comm_model.user_power[_user_power_allocation[i]]):6.6f} - "
                  f"bw: {float(self.comm_model.user_bandwidth[_user_bandwidth[i]]):6.6f} - "
                  f"cpu_freq: {_user_cpu_freq[i]} - "
                  f"distance: {float(self.comm_model.user_distance[ind]):6.2f} - "
                  f"W: {self.comm_model.W[ind, _user_bandwidth[i], _rb_allocation[i], _user_power_allocation[i], _user_cpu_freq[i]]:6.6f}")
        print("-------------------------")

        print(f"_ind_selected_clients: {_ind_selected_clients}")
        print(f"_selected_clients: {_selected_clients}")
        print(f"_user_bandwidth: {_user_bandwidth}")
        print(f"_rb_allocation: {_rb_allocation}")
        print(f"_user_power_allocation: {_user_power_allocation}")
        print(f"_user_cpu_freq: {_user_cpu_freq}")
       
        return _ind_selected_clients, _selected_clients, _user_bandwidth, _rb_allocation, _user_power_allocation, _user_cpu_freq
