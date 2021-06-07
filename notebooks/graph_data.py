import torch
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from os import path
from random import randint
import pyreadr
from typing import Tuple

class GraphData:
    file_points_asymp = path.join("lv-ortho-modes", "data", "surface_points_ASYMP.RData")
    file_points_mi = path.join("lv-ortho-modes", "data", "surface_points_MI.RData")
    file_face = path.join("lv-ortho-modes", "data", "surface_face.RData")

    df_points_asymp = pyreadr.read_r(file_points_asymp)["X.ASYMP"]
    df_points_mi = pyreadr.read_r(file_points_mi)["X.MI"]
    df_faces = pyreadr.read_r(file_face)["faces"]
    
    points = 2523
    
    def __init__(self, train_x: int=200, train_y: int=200, test_x: int=100, test_y: int=100) -> None:
        self.split_into_train_test(train_x, train_y, test_x, test_y)
        self.normalize_data()
        self.create_loader()
        
    def _split_points(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:    
        # order: [ED-endo ED-epi ES-endo ES-epi]
        
        
        ed_endo = df.iloc[:, :self.points]
        ed_epi = df.iloc[:, self.points:self.points*2]
        es_endo = df.iloc[:, self.points*2:self.points*3]
        es_epi = df.iloc[:, self.points*3:self.points*4]
        
        return (ed_endo, ed_epi, es_endo, es_epi)
    
    def split_into_train_test(self, train_x: int, train_y: int, test_x: int, test_y: int) -> None:
    
        points = { "ed_endo": {}, "es_endo": {} }
        
        points["ed_endo"]["asymp"], _, points["es_endo"]["asymp"], _ = self._split_points(self.df_points_asymp)
        points["ed_endo"]["mi"], _, points["es_endo"]["mi"], _ = self._split_points(self.df_points_mi)
        
        transformer = FaceToEdge(True)
        
        self.train_samples = []
        self.test_samples = []

        faces = torch.tensor(self.df_faces.values - 1, dtype=torch.long).t().contiguous()
        
        def create_data(start: int, end: int, ed_endo: pd.DataFrame, 
                        es_endo: pd.DataFrame, y: bool, list_: list
                       ) -> None:
            for ed, es in zip(ed_endo.values[start:end], es_endo.values[start:end]):
                pos_ed = torch.tensor(ed.reshape(self.points//3, 3), dtype=torch.float)
                pos_es = torch.tensor(es.reshape(self.points//3, 3), dtype=torch.float)
                data = Data(pos=pos_ed, face=faces, y=y)
                data.x = torch.tensor(np.concatenate((pos_ed, pos_es), axis=1))
                list_.append(transformer(data))
                
        create_data(0, train_x, points["ed_endo"]["asymp"], points["es_endo"]["asymp"], 0, self.train_samples)
        create_data(0, train_y, points["ed_endo"]["mi"], points["es_endo"]["mi"], 1, self.train_samples)
        create_data(train_x, train_x + test_x, points["ed_endo"]["asymp"], points["es_endo"]["asymp"], 0, self.test_samples)
        create_data(train_y, train_y + test_y, points["ed_endo"]["mi"], points["es_endo"]["mi"], 1, self.test_samples)
        
    def normalize_data(self) -> None:
        input_features = torch.cat([data.x for data in self.train_samples], axis=0)
        inputs_mean, inputs_std = torch.mean(input_features), torch.std(input_features)
        
        def normalize_set(dataset: list) -> None:
            for data in dataset:
                data.x = ((data.x - inputs_mean) / inputs_std)
            return dataset
                
        self.train_samples = normalize_set(self.train_samples)
        self.test_samples = normalize_set(self.test_samples)
        
    def create_loader(self) -> None:
        self.train_loader = DataLoader(self.train_samples, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_samples, batch_size=16, shuffle=True)
        
    def return_random_graph(self) -> Data:
        n = randint(0, len(self.test_samples))
        return self.test_samples[n]
