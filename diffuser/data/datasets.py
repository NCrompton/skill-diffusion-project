import minari
from .d4rl import suppress_output

class DatasetLoader():
    def __init__(self, env="D4RL/kitchen"):
        self.env = env

    def get_dataset(self, type):
        self.type = type
        self.dataset_name = f"{self.env}/{type}"

        # with suppress_output:
        self.dataset = minari.load_dataset(self.dataset_name, download=True)

        if self.dataset == None:
            raise DatasetNotFoundException(self.env, self.type)
        
        return self.dataset
    
    def get_env(self):
        if self.dataset == None:
            raise DatasetNotFoundException(self.env)
        
        return self.get_dataset().recover_environment()
        

class DatasetNotFoundException(Exception):
    def __init__(self, name, type):
        if type == None:
            super().__init__(f"dataset {name}/{type} is not found")
        else:
            super().__init__(f"dataset for {name} is not found")
