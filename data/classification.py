import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import pydicom
from dataclasses import dataclass
import pandas as pd
from enums.contrast import ClassificationContrasts
import os
from diskcache import Cache
import torchio as tio
import pickle



class ClassificationDataset(Dataset):
    def __init__(self, full_augment: bool, load_pickled: bool = False,) -> None:
        self.basepath = "/data/classification/"
        self.picklepath = "class_data/"
        targets = pd.read_csv(f"{self.basepath}train_labels.csv")
        targets["ID"] = targets["ID"].apply(lambda x: str(x).zfill(5))
        self._targets = targets.set_index("ID",)['MGMT_value'].to_dict()
        self._load_pickled = load_pickled
        self.full_augment = full_augment
        train_path = os.path.join(self.basepath, 'train/')
        self.candidates = os.walk(train_path).__next__()[1]
        
        transformations = []
        transformations.extend([tio.ToCanonical(),
            tio.Resample(tio.ScalarImage(os.path.join(train_path, self.candidates[0], ClassificationContrasts.flair.value))), 
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RescaleIntensity((0, 1)),
            tio.CropOrPad((112, 112, 64))
        ])
        augmentations = []
        if self.full_augment:
            augmentations.append(tio.OneOf(
                {tio.RandomAffine(): 0.8,
                 tio.RandomElasticDeformation(): 0.2
                 },
                p=0.75))

        self.__transforms = tio.Compose(transformations)
        self.__augmentation = tio.Compose(augmentations)
        
    def __len__(self) -> int:
        return len(self.candidates)  
    
    def load_candidate(self, index: int) -> tio.Subject:
        if self._load_pickled:
            with open(self.picklepath + str(index), "rb") as f:
                return pickle.load(f)
        candidate = self.candidates[index]
        path = os.path.join(self.basepath, 'train/', candidate)
        
        images = {}
        for contrast in ClassificationContrasts.values():
            image_path = os.path.join(path, contrast)
            images[contrast] = tio.ScalarImage(image_path)
        # images['label'] = one_hot(torch.as_tensor(self._targets[candidate]), num_classes=2)
        images['label'] = torch.tensor(self._targets[candidate])
        subject = tio.Subject(**images) 
        return subject

    def __getitem__(self, index: int) -> tio.Subject:
        candidate = self.load_candidate(index)
        candidate.load()
        if not self._load_pickled:
            candidate = self.__transforms(candidate)
        if self.full_augment:
            candidate = self.__augmentation(candidate)
        return candidate