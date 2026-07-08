import yaml
from pathlib import Path

class load_config:
    '''Loads the config file'''
    def __init__(self, path: str):
        self.path = path
    
    def load(self):
        with open(file = self.path, encoding= 'utf-8', mode = 'r') as yaml_file:
            return yaml.safe_load(yaml_file)