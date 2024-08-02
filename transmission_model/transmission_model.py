from typing import Any

import numpy as np


class Transmission_Model:

    def __init__(self, rb_number, user_number, total_model_params, lower_limit_distance=10, upper_limit_distance=500,
                 fixed_user_power=0, fixed_user_bw=0):

        self.rb_number = rb_number
        self.user_number = user_number
        self.total_model_params = total_model_params

        self.fixed_user_power = fixed_user_power
        self.fixed_user_bw = fixed_user_bw

        self.lower_limit_distance = lower_limit_distance
        self.upper_limit_distance = upper_limit_distance

        self.data_size_model = 0
        self.N = 10 ** -20
        self.q = np.array([])
        self.h = np.array([])

        self.user_power = np.array([])
        self.user_bandwidth = np.array([])
        self.user_interference = np.array([])
        self.user_distance = np.array([])
        self.user_angles = np.array([])
        #
        self.user_sinr = np.array([])
        self.user_data_rate = np.array([])
        self.user_delay = np.array([])

        self.base_station_power = 1  # W -> 30dBm
        self.base_station_bandwidth = 20  # MHz
        #
        self.base_station_sinr = np.array([])
        self.base_station_data_rate = np.array([])
        self.base_station_delay = np.array([])

        self.total_delay = np.array([])

        self.energy_coeff = 10 ** (-27)
        self.cpu_cycles = 40
        self.cpu_freq = 10 ** 9
        self.user_energy_training = np.array([])
        self.user_upload_energy = np.array([])
        self.total_energy = np.array([])

        print(f"> total_model_params: {total_model_params}")
        self.init()

    def init(self):
        self.init_user_interference()
        self.init_distance()
        self.init_user_power()
        self.init_user_bandwidth()
        self.init_q()
        self.init_h()
        self.init_user_sinr()
        self.init_user_data_rate()
        self.init_base_station_sinr()
        self.init_base_station_data_rate()
        self.init_data_size_model()
        self.init_user_delay()
        self.init_base_station_delay()
        self.init_totaldelay()
        self.init_user_energy_training()
        self.init_user_upload_energy()
        self.init_total_energy()

    def init_user_interference(self):
        i = np.array([0.05 + i * 0.01 for i in range(self.rb_number)])
        self.user_interference = (i - 0.04) * 0.000001

    def init_distance(self):
        np.random.seed(1)
        self.user_distance, self.user_angles = self.lower_limit_distance + (
                self.upper_limit_distance - self.lower_limit_distance) * np.random.rand(self.user_number,
                                                                                        1), 2 * np.pi * np.random.rand(
            self.user_number)
        np.random.seed()

    def init_user_power(self):
        if self.fixed_user_power == 0:
            # [0.008   0.00825 0.0085  0.00875 0.009   0.00925 0.0095  0.00975
            # 0.01
            #  0.01025 0.0105  0.01075 0.011   0.01125 0.0115  0.01175 0.012  ]
            inc = 0.00025
            self.user_power = np.arange(0.008, 0.012 + inc, inc)
        else:
            self.user_power = np.arange(self.fixed_user_power, 2 * self.fixed_user_power, self.fixed_user_power)

        print(f"user_power: {self.user_power}")

    def init_user_bandwidth(self):
        if self.fixed_user_bw == 0:
            inc = 0.1
            self.user_bandwidth = np.arange(1, 2 + inc, inc)
        else:
            self.user_bandwidth = np.arange(self.fixed_user_bw, 2 * self.fixed_user_bw, self.fixed_user_bw)

        print(f"user_bandwidth: {self.user_bandwidth}")

    def init_q(self):
        nmr = -1.08 * (self.user_interference + self.N * self.user_bandwidth[:, np.newaxis])
        dnr = (self.user_power * (self.user_distance ** -2))

        self.q = []
        i = 0
        for i_dnr in dnr:
            self.q.append([])
            for i_nmr in nmr:
                self.q[i].append(1 - np.exp(i_nmr[:, np.newaxis] / i_dnr))
            i = i + 1

        self.q = np.array(self.q)

    def init_h(self):
        o = 1  # Rayleigh fading parameter
        self.h = o * (self.user_distance ** (-2))

    def init_user_sinr(self):
        nmr = (self.user_power * self.h)[:, np.newaxis]
        dnr = (self.user_interference + self.user_bandwidth[:, np.newaxis] * self.N)

        self.user_sinr = []
        i = 0
        for i_nmr in nmr:
            self.user_sinr.append([])
            for i_dnr in dnr:
                self.user_sinr[i].append(i_nmr / i_dnr[:, np.newaxis])
            i = i + 1

        self.user_sinr = np.array(self.user_sinr)

    def init_user_data_rate(self):
        nmr = np.log2(1 + self.user_sinr)
        for i, _ in enumerate(nmr):
            for j, _ in enumerate(nmr[i]):
                nmr[i][j] = nmr[i][j] * self.user_bandwidth[j]

        self.user_data_rate = np.array(nmr)

    def init_base_station_sinr(self):
        base_station_interference = 0.06 * 0.000003  # Interference over downlink
        self.base_station_sinr = (self.base_station_power * self.h /
                                  (base_station_interference + self.N * self.base_station_power))

    def init_base_station_data_rate(self):
        self.base_station_data_rate = self.base_station_bandwidth * np.log2(1 + self.base_station_sinr)

    def init_data_size_model(self):
        # MBytes
        self.data_size_model = self.total_model_params * 4 / (1024 ** 2)

    def init_user_delay(self):
        self.user_delay = self.data_size_model / self.user_data_rate

    def init_base_station_delay(self):
        self.base_station_delay = self.data_size_model / self.base_station_data_rate

    def init_totaldelay(self):
        # self.total_delay = self.user_delay + self.base_station_delay
        self.total_delay = []
        for i, _ in enumerate(self.user_delay):
            self.total_delay.append([])
            self.total_delay[i].append(self.user_delay[i] + self.base_station_delay[i])

        self.total_delay = np.array(self.total_delay)

    def init_user_energy_training(self):
        self.user_energy_training = self.energy_coeff * self.cpu_cycles * (self.cpu_freq ** 2) * self.data_size_model

    def init_user_upload_energy(self):
        self.user_upload_energy = self.user_power * self.user_delay

    def init_total_energy(self):
        self.total_energy = self.user_energy_training + self.user_upload_energy
