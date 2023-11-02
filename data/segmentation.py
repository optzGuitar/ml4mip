import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from enums.contrast import Contrasts
from diskcache import Cache

class SegmentationDataset(Dataset):
    def __init__(self, full_augment: bool, use_cache: bool = False, cache_dir = './cache/') -> None:
        self.full_augment = full_augment
        self.basepath = "data/segmentation/train/"
        self.candidates = os.walk(self.basepath).__next__()[1]
    
        self._use_cache = use_cache
        self._cache = Cache(directory=cache_dir)
    
    def __len__(self):
        return len(self.candidates)
    
    def load_candidate(self, candidate: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(self.basepath, candidate)
        images = []
        
        for contrast in Contrasts.values():
            image = torch.as_tensor(
                nib.load(
                    os.path.join(path, contrast + "/", f"{candidate}_{contrast.lower()}.nii.gz")
                    ).get_fdata(), 
                dtype=torch.float
            )
            images.append(image)
            
        label = torch.as_tensor(nib.load(os.path.join(path, contrast + "/", f"{candidate}_seg.nii.gz")).get_fdata(), dtype=torch.float)
            
        if self._use_cache:
            self._cache[candidate] = torch.stack(images), label
            
        return torch.stack(images), label
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = self.candidates[index]
        
        if self._use_cache and candidate in self._cache:
            return self._cache[candidate]
        
        return self.load_candidate(candidate)