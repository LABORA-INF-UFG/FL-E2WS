import numpy as np
import pulp as pl
import re


class Communication_Strategy:

    def __init__(self, transmission_model, min_fit_clients, clients_number_data_samples, max_bandwidth,
                 delay_requirement=0.5, energy_requirement=0.003,
                 error_rate_requirement=0.3, lmbda=1.2):

        self.tm = transmission_model
        self.min_fit_clients = min_fit_clients
        self.max_bandwidth = max_bandwidth
        self.delay_requirement = delay_requirement
        self.energy_requirement = energy_requirement
        self.error_rate_requirement = error_rate_requirement
        self.lmbda = lmbda

        self.clients_number_data_samples = clients_number_data_samples

        self.count_selected_clients = 0
        self.selected_clients = np.array([])
        self.rb_allocation = np.array([])
        self.user_power_allocation = np.array([])
        self.user_bw_allocation = np.array([])

        self.success_uploads = []
        self.error_uploads = []

        self.W = np.array([])

        self.round_costs_list = {
            'total_training': [],
            'total_uploads': [],
            'total_error_uploads': [],
            'energy_success': [],
            'energy_error': [],
            'total_energy': [],
            'delay': [],
            'bw': [],
            'power': []
        }

        self.init()

    def init(self):
        self.compute_transmission_probability_matrix()

    def greater_data_user_selection(self, factor, k):

        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        data_samples_list = np.array(self.clients_number_data_samples)[selected_clients]
        pos_list = np.arange(len(data_samples_list))
        print(data_samples_list)

        combined_data = list(zip(data_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        distance_list, pos_list = zip(*sorted_data)

        print(f"data_list: {distance_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def greater_loss_user_selection(self, clients_loss_list, factor, k):
        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        loss_samples_list = np.array(clients_loss_list)[selected_clients]
        pos_list = np.arange(len(loss_samples_list))
        print(loss_samples_list)

        combined_data = list(zip(loss_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        loss_list, pos_list = zip(*sorted_data)

        print(f"data_list: {loss_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def random_user_selection(self, k):
        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[np.random.permutation(self.tm.user_number)[:k]] = 1
        self.selected_clients = np.where(self.selected_clients > 0)[0]
        self.count_selected_clients = len(self.selected_clients)

    def random_rb_allocation(self):
        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        self.rb_allocation[np.random.permutation(self.tm.rb_number)[:self.min_fit_clients]] = 1
        self.rb_allocation = np.random.permutation(np.where(self.rb_allocation > 0)[0])

    def fixed_user_power_allocation(self):
        self.user_power_allocation = np.zeros(self.count_selected_clients).astype(int)

    def fixed_user_bw_allocation(self):
        self.user_bw_allocation = np.zeros(self.count_selected_clients).astype(int)

    def compute_transmission_probability_matrix(self):
        self.W = np.zeros(
            (self.tm.user_number, len(self.tm.user_bandwidth), self.tm.rb_number, len(self.tm.user_power)))
        for i in range(self.tm.user_number):
            for j in range(len(self.tm.user_bandwidth)):
                for k in range(self.tm.rb_number):
                    for l in range(len(self.tm.user_power)):
                        if (self.tm.user_delay[i, j, k, l] < self.delay_requirement and  # total_delay
                                self.tm.user_upload_energy[i, j, k, l] < self.energy_requirement and  # total_energy
                                self.tm.q[i, j, k, l] <= self.error_rate_requirement):
                            self.W[i, j, k, l] = 1 - self.tm.q[i, j, k, l]

    def print_values(self):
        print("----------------------------------")
        print(f"selected_clients: {self.selected_clients}")
        if len(self.selected_clients) > 0:
            print(
                f"user_bw_allocation: {np.array(self.tm.user_bandwidth)[self.user_bw_allocation]} - Ind: {self.user_bw_allocation}")
            print(f"rb_allocation: {self.rb_allocation}")
            print(
                f"user_power_allocation: {np.array(self.tm.user_power)[self.user_power_allocation]} - Ind: {self.user_power_allocation}")
        print("----------------------------------")

    def upload_status(self):
        prob = np.random.rand(self.count_selected_clients)

        self.success_uploads = []
        self.error_uploads = []

        for i, ue in enumerate(self.selected_clients):
            prob_w = self.W[ue, self.user_bw_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]
            print(
                f"{i} - {ue} --> W: {prob_w:.4f} - P: {prob[i]:.4f} {'' if prob_w > 0 and prob_w >= prob[i] else ' - [X]'}")

            if prob_w > 0 and prob_w >= prob[i]:
                self.success_uploads.append(ue)
            else:
                self.error_uploads.append(ue)

    def round_costs(self):

        total_training = len(self.selected_clients)
        total_uploads = len(self.success_uploads)

        round_energy = 0
        round_energy_success = 0
        round_energy_error = 0
        round_delay = 0
        round_bw = 0
        round_power = 0

        for i, ue in enumerate(self.selected_clients):
            round_bw = round_bw + self.tm.user_bandwidth[self.user_bw_allocation[i]]
            round_power = round_power + self.tm.user_power[self.user_power_allocation[i]]
            round_energy = round_energy + self.tm.total_energy[
                ue, self.user_bw_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]
            round_delay = (round_delay + self.tm.total_delay[
                ue, 0, self.user_bw_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]])

            if ue in self.success_uploads:
                round_energy_success = round_energy_success + self.tm.total_energy[
                    ue, self.user_bw_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]
            else:
                round_energy_error = round_energy_error + self.tm.total_energy[
                    ue, self.user_bw_allocation[i], self.rb_allocation[i], self.user_power_allocation[i]]

        self.round_costs_list['total_training'].append(total_training)
        self.round_costs_list['total_uploads'].append(total_uploads)
        self.round_costs_list['total_error_uploads'].append(total_training - total_uploads)
        self.round_costs_list['energy_success'].append(round_energy_success)
        self.round_costs_list['energy_error'].append(round_energy_error)
        self.round_costs_list['total_energy'].append(round_energy)
        self.round_costs_list['delay'].append(round_delay)
        self.round_costs_list['bw'].append(round_bw)
        self.round_costs_list['power'].append(round_power)

    def print_round_costs(self):
        print("------------------------------------")
        print(f"total_training: {self.round_costs_list['total_training'][-1]}")
        print(
            f"total_uploads: {self.round_costs_list['total_uploads'][-1]}/{self.round_costs_list['total_training'][-1]}")
        print(
            f"total_error_uploads: {(self.round_costs_list['total_training'][-1] - self.round_costs_list['total_uploads'][-1])}/{self.round_costs_list['total_training'][-1]}")
        print("------------------------------------")

    def optimization(self):
        print(f"> optimization_rb_allocation")
        selected_clients = self.selected_clients.copy()

        print(f"len(selected_clients): {len(selected_clients)}")
        print(f"rb_number: {self.tm.rb_number}")
        print(f"min_fit_clients: {self.min_fit_clients}")

        print(f"delay_req: {self.delay_requirement}")
        print(f"energy_req: {self.energy_requirement}")
        print(f"error_rate_requirement: {self.error_rate_requirement}")
        print(f"selected_clients: {selected_clients}")
        print("*****************")

        # Creation of the assignment problem
        model = pl.LpProblem("Min_user_power_delay", pl.LpMinimize)

        # Decision Variables
        x = [[[[pl.LpVariable(f"x_{i}_{j}_{k}_{l}", cat=pl.LpBinary) for l in range(len(self.tm.user_power))] for k in
               range(self.tm.rb_number)] for j in range(len(self.tm.user_bandwidth))] for i in
             range(len(selected_clients))]

        # Objective function
        model += (pl.lpSum(self.tm.user_power[l] * self.tm.user_delay[selected_clients[i]][j][k][l] * x[i][j][k][l]
                           for l in range(len(self.tm.user_power)) for k in range(self.tm.rb_number)
                           for j in range(len(self.tm.user_bandwidth)) for i in range(len(selected_clients))) -
                  self.lmbda *
                  pl.lpSum(x[i][j][k][l]
                           for l in range(len(self.tm.user_power)) for k in range(self.tm.rb_number)
                           for j in range(len(self.tm.user_bandwidth))
                           for i in range(len(selected_clients))), "Min")

        model += pl.lpSum(
            x[i][j][k][l] for l in range(len(self.tm.user_power)) for k in range(self.tm.rb_number) for j in
            range(len(self.tm.user_bandwidth)) for i
            in range(len(selected_clients))) <= self.min_fit_clients, f"min_fit_clients"

        # Constraints: Each customer is assigned to exactly one bw, channel and power
        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][j][k][l] for l in range(len(self.tm.user_power))
                              for k in range(self.tm.rb_number) for j in
                              range(len(self.tm.user_bandwidth))) >= 0, f"constraint_client_channel_{i} >= 0"

        for i in range(len(selected_clients)):
            model += pl.lpSum(
                x[i][j][k][l] for l in range(len(self.tm.user_power)) for k in range(self.tm.rb_number) for j in
                range(len(self.tm.user_bandwidth))) <= 1, f"constraint_client_channel_{i} <= 1"

        # Constraints: Each channel is assigned to exactly one customer
        for k in range(self.tm.rb_number):
            model += pl.lpSum(
                x[i][j][k][l] for l in range(len(self.tm.user_power)) for i in range(len(selected_clients)) for j in
                range(len(self.tm.user_bandwidth))) >= 0, f"constraint_channel_client_{k} >= 0"

        for k in range(self.tm.rb_number):
            model += pl.lpSum(
                x[i][j][k][l] for l in range(len(self.tm.user_power)) for i in range(len(selected_clients)) for j in
                range(len(self.tm.user_bandwidth))) <= 1, f"constraint_channel_client_{k} <= 1"

        for i in range(len(selected_clients)):
            for j in range(len(self.tm.user_bandwidth)):
                for k in range(self.tm.rb_number):
                    for l in range(len(self.tm.user_power)):
                        # total_delay # total_energy
                        model += x[i][j][k][l] * self.tm.user_delay[selected_clients[i]][j][k][
                            l] <= self.delay_requirement, f"constraint_delay_{i}_{j}_{k}_{l}"
                        model += x[i][j][k][l] * self.tm.user_upload_energy[selected_clients[i]][j][k][
                            l] <= self.energy_requirement, f"constraint_energy_{i}_{j}_{k}_{l}"
                        model += x[i][j][k][l] * self.tm.q[selected_clients[i]][j][k][
                            l] <= self.error_rate_requirement, f"constraint_packet_error_rate_{i}_{j}_{k}_{l}"

        model += pl.lpSum((np.tile(np.repeat(self.tm.user_bandwidth, (len(self.tm.user_power) * self.tm.rb_number)),
                                   len(selected_clients)).reshape(
            (len(selected_clients), len(self.tm.user_bandwidth), self.tm.rb_number, len(self.tm.user_power)))[i][j][k][
            l]) * x[i][j][k][l] for l in
                          range(len(self.tm.user_power)) for k in range(self.tm.rb_number) for j in
                          range(len(self.tm.user_bandwidth)) for i in
                          range(len(selected_clients))) <= self.max_bandwidth, f"Bandwidth Budget"

        ################
        # Solving the problem
        status = model.solve()
        print(pl.LpStatus[status])
        print("Total cost:", pl.value(model.objective))

        self.selected_clients = []
        self.user_bw_allocation = []
        self.rb_allocation = []
        self.user_power_allocation = []
        for var in model.variables():
            if pl.value(var) == 1:
                indices = [int(i) for i in re.findall(r'\d+', var.name)]
                self.selected_clients.append(selected_clients[indices[0]])
                self.user_bw_allocation.append(indices[1])
                self.rb_allocation.append(indices[2])
                self.user_power_allocation.append(indices[3])

        print("<<<<<<<<<")
        print(self.selected_clients)
