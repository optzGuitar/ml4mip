import torch
from torch.utils.data import Dataset
import pydicom
from dataclasses import dataclass
import pandas as pd
from enums.contrast import Contrasts
import os


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
    def __init__(self) -> None:
        self.path = "data/classification/"
        targets = pd.read_csv(f"{self.path}train_labels.csv")
        self._targets = targets.set_index("ID", inplace=True)['MGMT_value'].to_dict()
        
        self.candidates = os.walk(os.path.join(self.path, 'train/')).__next__()[1]
        
    def load_candidate(self, candidate: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(self.path, 'train/', candidate)
        
        stacked_images = []
        for contrast in Contrasts.values():
            images: list[BrainSlice] = []
            
            for slice in os.walk(os.path.join(path, contrast)).__next__()[2]:
                images.append(BrainSlice.from_file(os.path.join(path, contrast, slice)))
                
            images.sort()
            stacked_images.append(torch.stack([image.brain_slice for image in images]))
            
        label = torch.as_tensor(self._targets[candidate])
            
        return torch.stack(images), torch.as_tensor([label], dtype=torch.float)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        candaidate = self.candidates[index]
        
        return self.load_candidate(candaidate)