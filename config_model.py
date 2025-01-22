from dataclasses import dataclass
from typing import List, Literal
import yaml

ActivationType = Literal['relu', 'sigmoid', 'tanh']

@dataclass
class CommonConfig:
    seed: int = 114514
    gpu: str = '0'
    runs: int = 10

@dataclass
class DatasetConfig:
    layers: List[int]
    acts: List[ActivationType]
    learning_rate: float
    pretrain_learning_rate: float
    lamSC: int
    coeff_reg: float
    max_epoch: int
    max_iter: int
    pre_iter: int

@dataclass
class CompleteConfig:
    common: CommonConfig
    acm: DatasetConfig
    wiki: DatasetConfig
    citeseer: DatasetConfig
    dblp: DatasetConfig
    film: DatasetConfig
    cornell: DatasetConfig
    cora: DatasetConfig
    wisc: DatasetConfig
    uat: DatasetConfig
    amap: DatasetConfig


    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CompleteConfig':
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(
            common=CommonConfig(**data['common']),
            acm=DatasetConfig(**data['acm']),
            wiki=DatasetConfig(**data['wiki']),
            citeseer=DatasetConfig(**data['citeseer']),
            dblp=DatasetConfig(**data['dblp']),
            film=DatasetConfig(**data['film']),
            cornell=DatasetConfig(**data['cornell']),
            cora=DatasetConfig(**data['cora']),
            wisc=DatasetConfig(**data['wisc']),
            uat=DatasetConfig(**data['uat']),
            amap=DatasetConfig(**data['amap'])
        )
