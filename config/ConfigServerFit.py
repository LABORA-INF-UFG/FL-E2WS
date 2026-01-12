class ConfigServerFit:
    def __init__(self, ds_clients, min_fit_clients, rb_number,
                 delay_requirement, 
                 error_rate_requirement, lmbda, beta, gama, fixed_parameters,
                 user_power, user_bw, user_cpu_freq, optmizer_type, cm):
        self.ds_clients = ds_clients
        self.min_fit_clients = min_fit_clients
        self.rb_number = rb_number

        self.delay_requirement = delay_requirement
        self.error_rate_requirement = error_rate_requirement

        self.lmbda = lmbda
        self.beta = beta
        self.gama = gama

        self.fixed_parameters = fixed_parameters

        self.user_power = user_power
        self.user_bw = user_bw
        self.user_cpu_freq = user_cpu_freq

        self.optmizer_type = optmizer_type

        self.cm = cm