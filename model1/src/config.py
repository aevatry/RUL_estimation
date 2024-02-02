import os
from tkinter import filedialog
import json


class Config:
    def __init__(self,config_path:str) -> None:

        """
        Get the model configuration from the JSON file and asks for 
        """
        
        self.config_path = config_path

        with open(config_path, 'r') as f:
            config_dict = json.loads(f.read())
        
        # Get the previous config file parameter field names 
        self.config_keys = list(config_dict.keys())
        self.load_attributes(**config_dict)

        # Select the correct directories
        print("Choose TRAIN directory\n")
        self._train_dir = filedialog.askdirectory(**{'mustexist':True}) 
        assert len(self._train_dir)>0, "You need a training directory, even if empty"
        print("Choose EVAL directory\n")       
        self._eval_dir = filedialog.askdirectory(**{'initialdir': os.path.dirname(self._train_dir),'mustexist':True})
        assert len(self._eval_dir)>0, "You need an eval directory, even if empty"


    def load_attributes(self, **attrs):

        self.__dict__.update(**attrs)

    def save(self):

        new_dict = {k:self.__dict__[k] for k in self.config_keys}

        with open(self.config_path, 'w') as f:
            f.write(json.dumps(new_dict))
