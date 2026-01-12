class ConfigModel:
    def __init__(self, model_type="MLP", shape=(28, 28, 1),
                 path_clients="", path_server=""):

        self.model_type = model_type
        self.shape = shape
        self.path_clients = path_clients
        self.path_server = path_server