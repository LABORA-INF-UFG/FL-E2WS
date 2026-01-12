import sys
import tensorflow as tf
from client.GenerateClientList import GenerateClientList
from config.ConfigModel import ConfigModel
from config.ConfigServerFit import ConfigServerFit
from network_topologies.cloud_server.ServerStrategyFit import ServerStrategyFit

if __name__ == "__main__":
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Device: {device}")

    if device == '/CPU:0':        
        sys.exit()

    for i in range(15):
        with tf.device(device):
            ds_clients = GenerateClientList(n_es=1, n_device=150, lower_limit=100, upper_limit=500)

            csf = ConfigServerFit(ds_clients=ds_clients,
                                  min_fit_clients=10,
                                  rb_number=15,                         

                                  delay_requirement=0.3,                                  
                                  error_rate_requirement=0.3,

                                  lmbda=6, 
                                  beta=1.4, 
                                  gama=2.1, 

                                  fixed_parameters=False,

                                  user_power=0.01,
                                  user_bw=1,
                                  user_cpu_freq=1,

                                  optmizer_type = "E2WS",
                                  # optmizer_type="FLoWN",
                                  # optmizer_type="FedAvg",
                                  # optmizer_type="FedProx",

                                  cm=ConfigModel(model_type="CNN", # model_type="MLP" / model_type="CNN"
                                                 shape=(28, 28, 1),                                   
                                                 path_clients="",
                                                 path_server=""
                                                 )
                                )
            
            s_stf = ServerStrategyFit(sid=0, n_rounds=100, csf=csf)

            for i_round in range(s_stf.n_rounds):
                print("*************************")
                print(f"ServerRound: {i_round + 1}")
                s_stf.fit()

            s_stf.print_result()


