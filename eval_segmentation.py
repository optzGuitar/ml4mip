import torch
from torch.utils.data import DataLoader
from data.segmentation_data import SegmentationDataset
from segmentation.config import DataConfig, LossConfig, SegmentationConfig, TrainConfig
from segmentation.seg_module import SegModule
import nibabel as nib
import torchio as tio

if __name__ == "__main__":
    config = config = SegmentationConfig(
        run_name="tio_pipeline_fr",
        train_config=TrainConfig(
            epochs=50,
            gradient_accumulation_steps=16,
            batch_size=4,
        ),
        loss_config=LossConfig(
            cosine_period=200
        ),
        data_config=DataConfig(load_pickle=False)
    )

    seg_dataset = SegmentationDataset(
        config=config, full_augment=False, load_test=True
    )
    seg_loader = DataLoader(seg_dataset, batch_size=2, shuffle=False)

    module = SegModule.load_from_checkpoint(
        f"/home/tu-leopinetzki/ml4mip/segmentation_checkpoints/{config.run_name}_last.ckpt", config=config)

    def split_batch(self, batch):
        images = torch.concat(
            [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        return images, batch["FLAIR"][tio.AFFINE]

    module.unet.eval()
    with torch.no_grad():
        for i, batch in enumerate(seg_loader):
            x, affine = split_batch(batch)

            seg = module.unet(x)

            for n, segmentation in enumerate(seg):
                idx = (i * config.train_config.batch_size) + n
                nib.save(
                    nib.Nifti1Image(
                        segmentation.cpu().numpy(),
                        affine=affine.cpu().numpy()
                    ),
                    f"/submission/hazel/segmentation/test_{idx}/test_{idx}_seg.nii.gz"
                )