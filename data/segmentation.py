from torch.utils.data import Dataset
import os
import nibabel as nib
from enums.contrast import Contrasts
from diskcache import Cache
import torchio as tio
# from monai.transforms import Compose, LoadImage,  EnsureChannelFirst, Orientation, Spacing,  NormalizeIntensity,  ScaleIntensity


class SegmentationDataset(Dataset):
    def __init__(self, full_augment: bool, num_classes: int, cache_dir='./cache/'):
        self.full_augment = full_augment
        self.basepath = "data/segmentation/train/"
        self.candidates = os.walk(self.basepath).__next__()[1]

        self._cache = Cache(directory=cache_dir)
        self._num_classes = num_classes

        self.__transforms = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RescaleIntensity((0, 1)),
            tio.OneOf(
                {tio.RandomAffine(): 0.8,
                 tio.RandomElasticDeformation(): 0.2
                 },
                p=0.75),
            tio.CropOrPad((224, 224, 128)),
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
        images['label'] = tio.OneHot(
            num_classes=self._num_classes)(images['label'])

        data = tio.Subject(**images)

        return data

    def __getitem__(self, index: int) -> tio.Subject:
        candidate = self.candidates[index]

        if index in self._cache:
            candidate = self._cache[index]
        else:
            candidate = self.load_candidate(candidate)
            candidate.load()
            self._cache[index] = candidate

        transformed_candidate = self.__transforms(candidate)
        return transformed_candidate
