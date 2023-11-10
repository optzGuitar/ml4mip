import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from enums.contrast import Contrasts
from diskcache import Cache
import torchio as tio
from monai.transforms import Compose, LoadImage,  EnsureChannelFirst, Orientation, Spacing,  NormalizeIntensity,  ScaleIntensity


class SegmentationDataset(Dataset):
    def __init__(self, full_augment: bool, use_cache: bool = False, cache_dir='./cache/'):
        self.full_augment = full_augment
        self.basepath = "data/segmentation/train/"
        self.candidates = os.walk(self.basepath).__next__()[1]

        self._use_cache = use_cache
        self._cache = Cache(directory=cache_dir)

        self.__transforms = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RescaleIntensity((0, 1)),
            tio.OneOf(
                {tio.RandomAffine(): 0.8,
                 tio.RandomElasticDeformation(): 0.2
                 },
                p=0.75),
            tio.EnsureShapeMultiple(target_multiple=(2, 2, 2)),
        ])

    def __len__(self) -> int:
        return len(self.candidates)

    def load_candidate(self, candidate: str) -> tio.Subject:
        path = os.path.join(self.basepath, candidate)
        images = []

        images = {}
        for contrast in Contrasts.values():
            image_path = os.path.join(
                path, f"{candidate}_{contrast.lower()}.nii.gz")

            images[contrast] = tio.ScalarImage(image_path)

        images['label'] = tio.LabelMap(os.path.join(
            path, f"{candidate}_seg.nii.gz"))

        data = tio.Subject(**images)

        return data

    def __getitem__(self, index: int) -> tio.Subject:
        candidate = self.candidates[index]

        if self._use_cache and index in self._cache:
            # TODO: also remove from cache after some time to allow new transformations
            return self._cache[candidate]

        candidate = self.load_candidate(candidate)
        candidate.load()

        transformed_candidate = self.__transforms(candidate)

        if self._use_cache:
            self._cache[index] = transformed_candidate

        return transformed_candidate
