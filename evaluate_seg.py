from data.segmentation_data import SegmentationDataset
from segmentation.config import DataConfig, LossConfig, SegmentationConfig, TrainConfig
from segmentation.seg_module import SegModule
from torch.utils.data import DataLoader, random_split
import torch

config = SegmentationConfig(
    run_name="final_unet",
    train_config=TrainConfig(
        epochs=75,
        gradient_accumulation_steps=16,
        batch_size=2,
        num_workers=7,
    ),
    loss_config=LossConfig(
        cosine_period=2175,
    ),
    data_config=DataConfig(load_pickle=True)
)

generator = torch.manual_seed(config.seed)

model = SegModule.load_from_checkpoint(
    "segmentation_checkpoints/final_unet.ckpt", segmentation_config=config
)

dataset = SegmentationDataset(
    config, True)

_, val_dataset = random_split(
    dataset, (0.95, 0.05), generator=generator
)

loader = DataLoader(
    val_dataset, batch_size=1, num_workers=config.train_config.num_workers
)

validation_results = []
for batch in loader:
    with torch.no_grad():
        x, y = model.split_batch(batch)
        x = x.float()

        prediction = model.unet(x)
        results = model.metrics(prediction, y, is_train=False)
        validation_results.append(results)

        torch.cuda.empty_cache()

accumulated = {}
for result in validation_results:
    for key, value in result.items():
        if key not in accumulated:
            accumulated[key] = 0
        accumulated[key] += value

for key, value in accumulated.items():
    print(f"{key}: {value / len(validation_results)}")
    accumulated[key] = value / len(validation_results)

with open("validation_results.txt", "w") as f:
    f.write(str(accumulated))
