import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianAlgorithmOpt:

    @staticmethod
    def opt(self):
        print(f"> Hungarian Algorithm")

        selected_clients = self.selected_clients    

        #  C[i,j] = q[i,j,k*]. If there are no viable powers, cost = 999.
        C = np.zeros((len(selected_clients), self.csf.rb_number))
        best_k = np.full((len(selected_clients), self.csf.rb_number), -1, dtype=int)

        for i in range(len(selected_clients)):
            idx = i # selected_clients[i]
            for j in range(self.csf.rb_number):
                # Lists viable powers for (i, j)
                feasible_powers = []
                for k in range(len(self.comm_model.user_power)):
                    if (#(self.cs.tm.user_upload_energy[idx, j, k] <= self.cs.energy_requirement and
                            self.comm_model.user_delay_upload[idx, 0, j, k] <= self.csf.delay_requirement and
                            self.comm_model.q[idx, 0, j, k] <= self.csf.error_rate_requirement):
                        feasible_powers.append(k)

                if not feasible_powers:
                    # No power meets the restrictions
                    C[i, j] = 999.0
                    best_k[i, j] = -1
                else:
                    # Choose the lowest power (Eq. (22)) => min user_p[k]
                    chosen_p = 9999.0
                    chosen_k = -1
                    for k2 in feasible_powers:
                        if self.comm_model.user_power[k2] < chosen_p:
                            chosen_p = self.comm_model.user_power[k2]
                            chosen_k = k2

                    C[i, j] = self.comm_model.q[idx, 0, j, chosen_k]
                    best_k[i, j] = chosen_k

        print(best_k)

        # Hungarian algorithm
        assigned_users_list = []
        not_assigned_list = []
        row_ind, col_ind = linear_sum_assignment(C)

        _rb_allocation = []
        _user_power_allocation = []
        _ind_selected_clients = []
        total_cost = 0.0
        for idx in range(len(row_ind)):
            i_user = row_ind[idx]
            j_rb = col_ind[idx]
            cost_val = C[i_user, j_rb]
            k_power = best_k[i_user, j_rb]
            total_cost += cost_val

            if k_power < 0:
                not_assigned_list.append(selected_clients[idx])
            else:
                pot_val = self.comm_model.user_power[k_power]
                assigned_users_list.append(selected_clients[idx])
                _rb_allocation.append(j_rb)
                _user_power_allocation.append(k_power)
                _ind_selected_clients.append(idx)
                print(f"User [{i_user}]: {selected_clients[idx]} -> (RB={j_rb}, Power={pot_val:.4f}), cost(q)={cost_val:.6f}, distance: {np.squeeze(self.csf.ds_clients.user_distance)[selected_clients[idx]-1]}") # , distance: {self.comm_model.user_distance[selected_clients[idx]]}

        _user_bandwidth = np.zeros(len(assigned_users_list), dtype=int).tolist()
        _user_cpu_freq = np.full(len(assigned_users_list), len(self.comm_model.cpu_freq)-1).tolist()

        print(f"\nFull cost = {total_cost:.6f}")
        print(f"assigned_users_list: {assigned_users_list}")
        print(f"not_assigned_list: {not_assigned_list}\n")

        print(f"_selected_clients: {assigned_users_list}")
        print(selected_clients)
        print(f"_rb_allocation: {_rb_allocation}")
        print(f"_user_power_allocation: {_user_power_allocation}")

        return _ind_selected_clients, assigned_users_list, _user_bandwidth, _rb_allocation, _user_power_allocation, _user_cpu_freq
