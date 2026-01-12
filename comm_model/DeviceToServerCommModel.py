import sys
import numpy as np


class DeviceToServerCommModel:

    def __init__(self, csf, total_model_params,
                 list_clients,
                 user_distance):

        self.csf = csf
        self.total_model_params = total_model_params
        print(f"total_model_params: {self.total_model_params}")

        self.data_size_model = 0

        self.list_clients = list_clients
        self.user_distance = np.array(user_distance).reshape(-1, 1)

        self.user_bandwidth = np.array([])
        self.N = 10 ** -20

        self.user_interference = np.array([])
        self.user_power = np.array([])
        self.q = np.array([])
        self.h = np.array([])

        self.user_sinr = np.array([])
        self.user_sinr_avg = {}
        self.user_data_rate = np.array([])
        self.user_delay_training = np.array([])
        self.user_delay_upload = np.array([])

        self.base_station_power = 1 # W
        self.base_station_bandwidth = 20  # MHz
        #
        self.base_station_sinr = np.array([])
        self.base_station_data_rate = np.array([])
        self.base_station_delay = np.array([])

        self.total_delay = np.array([])
        self.total_user_delay = np.array([])

        self.energy_coeff = 10 ** (-27)
        self.cpu_cycles = 40
        self.cpu_freq = np.array([])
        self.user_energy_training = np.array([])
        self.user_upload_energy = np.array([])
        self.total_energy = np.array([])
        self.W = np.array([])
        self.init()

    def init(self):
        self.init_user_interference()
        self.init_user_power()
        self.init_user_bandwidth()
        self.init_q()
        self.init_h()
        self.init_user_sinr()
        self.init_user_sinr_avg()
        self.init_user_data_rate()
        self.init_base_station_sinr()
        self.init_base_station_data_rate()
        self.init_data_size_model()

        self.init_cpu_freq()
        self.init_user_delay_training()
        self.init_user_delay_upload()

        self.init_base_station_delay()
        self.init_user_delay_training()
        self.init_user_energy_training()
        self.init_total_delay()
        self.init_total_user_delay()

        self.init_user_upload_energy()
        self.init_total_energy()
        self.init_compute_transmission_success()

    def init_user_interference(self):
        i = np.array([0.05 + i * 0.01 for i in range(self.csf.rb_number)])
        self.user_interference = (i - 0.04) * 0.000001

    def init_user_power(self):
        if self.csf.fixed_parameters is False:
            inc = self.csf.user_power / 20
            self.user_power = np.arange(self.csf.user_power/2, self.csf.user_power + inc, inc)
            self.user_power = self.user_power[self.user_power <= self.csf.user_power]
        else:
            self.user_power = np.array([self.csf.user_power])    

    def init_user_bandwidth(self):
        if self.csf.fixed_parameters is False:
            inc = self.csf.user_bw / 5
            self.user_bandwidth = np.arange(self.csf.user_bw, self.csf.user_bw * 2 + inc, inc)
            self.user_bandwidth = self.user_bandwidth[self.user_bandwidth <= self.csf.user_bw * 2]
        else:
            self.user_bandwidth = np.array([self.csf.user_bw])

    def init_cpu_freq(self):
        if self.csf.fixed_parameters is False:
            inc = self.csf.user_cpu_freq / 20
            self.cpu_freq = np.arange(self.csf.user_cpu_freq/2 - inc, self.csf.user_cpu_freq + inc, inc)[-8:]
        else:
            self.cpu_freq = np.array([self.csf.user_cpu_freq])
        self.cpu_freq = self.cpu_freq[self.cpu_freq <= self.csf.user_cpu_freq]
        self.cpu_freq = self.cpu_freq * 10 ** 9 

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
        o = 1
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

    def init_user_sinr_avg(self):
        for i, cid in enumerate(self.list_clients):
            self.user_sinr_avg[cid] = np.mean(self.user_sinr[i])

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
        self.data_size_model = self.total_model_params * 4 / (1024 ** 2)         

    def init_user_delay_training(self):
        self.user_delay_training = self.cpu_cycles * self.data_size_model / self.cpu_freq 

    def init_user_energy_training(self):
        self.user_energy_training = self.energy_coeff * self.cpu_cycles * (self.cpu_freq ** 2) * self.data_size_model
 
    def init_user_delay_upload(self):
        self.user_delay_upload = self.data_size_model / self.user_data_rate

    def init_base_station_delay(self):
        self.base_station_delay = self.data_size_model / self.base_station_data_rate

    def init_total_delay(self):
        delay_user_bs = []
        for i, _ in enumerate(self.user_delay_upload):
            delay_user_bs.append([])
            delay_user_bs[i].append(self.user_delay_upload[i] + self.base_station_delay[i])            

        delay_user_bs = np.array(delay_user_bs)
        self.total_delay = np.zeros(delay_user_bs.shape + (self.user_delay_training.shape[0],))
        for i in range(self.user_delay_training.shape[0]):
            self.total_delay[..., i] = delay_user_bs + self.user_delay_training[i]

    def init_total_user_delay(self):
        self.total_user_delay = np.zeros(self.user_delay_upload.shape + (self.user_delay_training.shape[0],))
        for i in range(self.user_delay_training.shape[0]):
            self.total_user_delay[..., i] = self.user_delay_upload + self.user_delay_training[i]

    def init_user_upload_energy(self):
        self.user_upload_energy = self.user_power * self.user_delay_upload

    def init_total_energy(self):
        self.total_energy = np.zeros(self.user_upload_energy.shape + (self.user_energy_training.shape[0],))
        for i in range(self.user_energy_training.shape[0]):
            self.total_energy[..., i] = self.user_upload_energy + self.user_energy_training[i]            

    def init_compute_transmission_success(self):
        self.W = np.zeros((len(self.list_clients), len(self.user_bandwidth), self.csf.rb_number, len(self.user_power), len(self.cpu_freq)))
        for i in range(len(self.list_clients)):
            for j in range(len(self.user_bandwidth)):
                for k in range(self.csf.rb_number):
                    for l in range(len(self.user_power)):
                        for m in range(len(self.cpu_freq)):                            
                            if (self.user_delay_upload[i, j, k, l] <= self.csf.delay_requirement and                                    
                                    self.q[i, j, k, l] <= self.csf.error_rate_requirement):
                                self.W[i, j, k, l, m] = 1 - self.q[i, j, k, l]



