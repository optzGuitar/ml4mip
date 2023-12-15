import pickle
from torch.utils.data import Dataset
import os
import torch
import nibabel as nib
from enums.contrast import Contrasts
from diskcache import Cache
import torchio as tio
# from monai.transforms import Compose, LoadImage,  EnsureChannelFirst, Orientation, Spacing,  NormalizeIntensity,  ScaleIntensity
import nibabel as nib
from segmentation.config import SegmentationConfig


class SegmentationDataset(Dataset):
    def __init__(self, config: SegmentationConfig, full_augment: bool, load_picked: bool = True):
        self.full_augment = full_augment
        self.basepath = "data/segmentation/train/"
        self.candidates = os.walk(self.basepath).__next__()[1]

        self.pickled_path = "group/hazel/seg_data/"
        self._load_pickled = load_picked

        transformations = []
        if not self._load_pickled:
            transformations.extend([tio.ZNormalization(
                masking_method=tio.ZNormalization.mean),
                tio.RescaleIntensity((0, 1)),
                tio.CropOrPad(config.data_config.image_size),
            ])
        if self.full_augment:
            transformations.append(tio.OneOf(
                {tio.RandomAffine(): 0.8,
                 tio.RandomElasticDeformation(): 0.2
                 },
                p=0.75))

        self.__transforms = tio.Compose(transformations)
        self.config = config

    def __len__(self) -> int:
        return len(self.candidates)

    def load_candidate(self, index: int) -> tio.Subject:
        if self._load_pickled:
            with open(f"{self.pickled_path}{index}.pkl", "rb") as file:
                return pickle.load(file)

        candidate = self.candidates[index]
        path = os.path.join(self.basepath, candidate)

        images = {}
        for contrast in Contrasts.values():
            image_path = os.path.join(
                path, f"{candidate}_{contrast.lower()}.nii.gz")
            tensor = torch.as_tensor(
                nib.load(image_path).get_fdata(), dtype=torch.float
            ).unsqueeze(0)
            images[contrast] = tio.ScalarImage(tensor=tensor)

        tensor = torch.as_tensor(
            nib.load(os.path.join(
                path, f"{candidate}_seg.nii.gz")
            ).get_fdata(), dtype=torch.float
        ).unsqueeze(0)
        labels = torch.stack((
            (tensor == 0)
            (tensor == 1)
            (tensor == 2),
            (tensor == 4),
        ))
        images['label'] = tio.LabelMap(tensor=labels)
        images['label'] = tio.OneHot(
            num_classes=self.config.data_config.n_classes)(images['label'])

        data = tio.Subject(**images)

        return data

    def __getitem__(self, index: int) -> tio.Subject:
        candidate = self.load_candidate(index)
        if not self._load_pickled:
            candidate.load()

        transformed_candidate = self.__transforms(candidate)
        return transformed_candidate
