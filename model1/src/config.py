import json

class Config:
    def __init__(self,config_path:str) -> None:
        
        self.config_path = config_path

        with open(config_path, 'r') as f:
            config_dict = json.loads(f.read())
        
        self.config_keys = list(config_dict.keys())
        self.load_attributes(**config_dict)

        

    def load_attributes(self, **attrs):

        self.__dict__.update(**attrs)

    def save(self):

        new_dict = {k:self.__dict__[k] for k in self.config_keys}

        with open(self.config_path, 'w') as f:
            f.write(json.dumps(new_dict))