# src/configs/config.py

import yaml
from typing import Any, Dict

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        self.data = ConfigDict(cfg.get('data', {}))
        self.preprocessing = ConfigDict(cfg.get('preprocessing', {}))
        self.training = ConfigDict(cfg.get('training', {}))
        self.model = ConfigDict(cfg.get('model', {}))

class ConfigDict:
    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            setattr(self, key, value)

