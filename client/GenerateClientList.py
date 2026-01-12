import sys
import numpy as np


class GenerateClientList:
    def __init__(self, n_es, n_device, lower_limit=100, upper_limit=500):
        self.n_es = n_es
        self.n_device = n_device
        self.user_number = self.n_es * self.n_device

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.group_list = []
        self.user_distance = []
        self.user_angles = []
        self.init_group_list()
        self.init_device_distance()

    def init_group_list(self):
        self.group_list = np.reshape(list(range(1, self.user_number + 1)), (self.n_es, self.n_device))

    def init_device_distance(self):
        np.random.seed(1)
        (user_distance,
         self.user_angles) = (self.lower_limit + (self.upper_limit - self.lower_limit) *
                              np.random.rand(self.user_number, 1),
                              2 * np.pi * np.random.rand(self.user_number))
        self.user_distance = np.reshape(user_distance, (self.n_es, self.n_device))
        np.random.seed()




