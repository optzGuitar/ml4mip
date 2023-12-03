from data.classification import ClassificationDataset
from torch.utils.data import DataLoader
from monai.networks.nets import resnet
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda import device_count
from classification.train_util import TrainLoop

dataset = ClassificationDataset(full_augment=False)
datal = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = resnet.resnet50(spatial_dims=3, num_classes=2)
optimizer = AdamW(model.parameters(), lr=1e-1, weight_decay=0.001)
epochs = 1


TrainLoop(
    model=model,
    dataloader=datal,
    loss_fn=CrossEntropyLoss,
    optimizer=optimizer
).loop(epochs)
