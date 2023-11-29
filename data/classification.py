import torch
from torch.utils.data import Dataset
import pydicom
from dataclasses import dataclass
import pandas as pd
from enums.contrast import ClassificationContrasts
import os
from diskcache import Cache
import torchio as tio

@dataclass
class BrainSlice:
    brain_slice: torch.Tensor
    id: int
    
    @classmethod
    def from_file(cls, path: str):
        return cls(
            brain_slice=torch.as_tensor(
                pydicom.dcmread(path).pixel_array
            ),
            id=int(path.split("/")[-1].split(".")[0].split("-")[-1])
        )
    
    def __lt__(self, other):
        if isinstance(other, BrainSlice):
            return self.id < other.id
        super().__lt__(other)
        
    def __gt__(self, other):
        if isinstance(other, BrainSlice):
            return self.id > other.id
        super().__gt__(other)
        
    def __eq__(self, other) -> bool:
        if isinstance(other, BrainSlice):
            return self.id == other.id
        super().__eq__(other)


class ClassificationDataset(Dataset):
    def __init__(self, full_augment: bool, use_cache: bool = False, cache_dir = './cache/') -> None:
        self.basepath = "/data/classification/"
        targets = pd.read_csv(f"{self.basepath}train_labels.csv")
        targets["ID"] = targets["ID"].apply(lambda x: str(x).zfill(5))
        self._targets = targets.set_index("ID",)['MGMT_value'].to_dict()

        self.full_augment = full_augment
        self.candidates = os.walk(os.path.join(self.basepath, 'train/')).__next__()[1]
        self._cache = Cache(directory=cache_dir)
        self._use_cache = use_cache
        
    def __len__(self) -> int:
        return len(self.candidates)  
    
    def load_candidate(self, candidate: str) -> tio.Subject:
        path = os.path.join(self.basepath, 'train/', candidate)
        
        images = {}
        for contrast in ClassificationContrasts.values():
            image_path =  image_path = os.path.join(path, contrast)
            images[contrast] = tio.ScalarImage(image_path)
            
        images['label'] = torch.as_tensor(self._targets[candidate])
        subject = tio.Subject(**images) 

        if self._use_cache:
            self._cache[candidate] = subject
        return subject
    
    def __len__(self) -> int:
        return len(self.candidates)
    
    def __getitem__(self, index: int) -> tio.Subject:
        candaidate = self.candidates[index]
        
        if self._use_cache and candaidate in self._cache:
            return self._cache[candaidate]
        
        return self.load_candidate(candaidate)