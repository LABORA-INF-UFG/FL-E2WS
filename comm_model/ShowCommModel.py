import numpy as np


class ShowCommModel:

    @staticmethod
    def show_q(comm_model, i=-1):
        print("-> q")
        if i == -1:
            print(comm_model.q)
        else:
            pass

    @staticmethod
    def show_w(comm_model, i=-1):
        print("-> W")
        if i == -1:
            print(comm_model.W)
        else:
            pass

    @staticmethod
    def show_total_user_delay(comm_model, i=-1):
        print("-> total_user_delay")
        if i == -1:
            print(comm_model.total_user_delay)
        else:
            pass

    @staticmethod
    def show_total_energy(comm_model, i=-1):
        print("-> total_energy")
        if i == -1:
            print(comm_model.total_energy)
        else:
            pass

    @staticmethod
    def show_user_sinr(comm_model, i=-1):
        print("-> sinr")
        if i == -1:
            print(comm_model.user_sinr[0])
        else:
            pass

    @staticmethod
    def show_user_sinr_avg(comm_model, i=-1):
        print("-> sinr_avg")
        if i == -1:
            print(comm_model.user_distance)
            print(comm_model.user_sinr_avg)
        else:
            pass


    @staticmethod
    def show(comm_model):
        pass


