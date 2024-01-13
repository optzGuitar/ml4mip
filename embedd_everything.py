import os
import pickle
from data.classification import EmbeddingDataset  # noqa
import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/classif")  # noqa
import torch
from classification.embedding import ResNet18
from data.classification import ClassificationDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    resnet = ResNet18(512, channels=1)
    resnet.load_state_dict(pickle.load("embedding_end.pkl"))

    resnet.eval()
    resnet.to("cuda:0")

    with torch.no_grad():
        dataset = ClassificationDataset(full_augment=False, load_pickled=True)
        loader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=4)
        for i, batch in enumerate(loader):
            data = batch.to(resnet.device)

            embedded = []
            slice = slice.permute(4, 0, 1, 2, 3)
            orig_shape = slice.shape
            slice = slice.view(
                orig_shape[0] * orig_shape[1], *orig_shape[2:])

            emb0 = resnet(slice[:, 0:1]).view(
                orig_shape[0], orig_shape[1], -1)
            emb1 = resnet(data[:, 1:2]).view(
                orig_shape[0], orig_shape[1], -1)
            emb2 = resnet(data[:, 2:3]).view(
                orig_shape[0], orig_shape[1], -1)
            emb3 = resnet(data[:, 3:4]).view(
                orig_shape[0], orig_shape[1], -1)

            emb0 = emb0.permute(1, 2, 0)
            emb1 = emb1.permute(1, 2, 0)
            emb2 = emb2.permute(1, 2, 0)
            emb3 = emb3.permute(1, 2, 0)

            for x, image in enumerate(torch.stack([emb0, emb1, emb2, emb3], dim=1)):
                embedded = image.cpu()
                with open(f"embeddings/embedding_{(i*16) + x}.pkl", "wb") as f:
                    pickle.dump(embedded, f)
