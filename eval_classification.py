

import os
import pickle

import torch
from classification.embedding import ResNet18
from classification.embedding_classifier import MGMTClassifier
from data.classification import ClassificationDataset, EmbeddedDataset, EmbeddingDataset
from torch.utils.data import DataLoader
import torchio as tio
import pandas as pd


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    resnet = ResNet18(512, channels=1)
    with open("embedding_end.pkl", "rb") as f:
        resnet.load_state_dict(pickle.load(f))

    resnet.eval()
    resnet = resnet.to("cuda:0")

    with torch.no_grad():
        dataset = ClassificationDataset(
            full_augment=False, load_pickled=True, load_test=True)
        loader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=4)
        for i, batch in enumerate(loader):
            input_images = torch.cat(
                [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)
            data = input_images.to("cuda:0")

            embedded = []
            slice = data.permute(4, 0, 1, 2, 3)
            orig_shape = slice.shape
            slice = slice.reshape(
                orig_shape[0] * orig_shape[1], *orig_shape[2:])

            emb0 = resnet(slice[:, 0:1]).reshape(
                orig_shape[0], orig_shape[1], -1)
            emb1 = resnet(slice[:, 1:2]).reshape(
                orig_shape[0], orig_shape[1], -1)
            emb2 = resnet(slice[:, 2:3]).reshape(
                orig_shape[0], orig_shape[1], -1)
            emb3 = resnet(slice[:, 3:4]).reshape(
                orig_shape[0], orig_shape[1], -1)

            emb0 = emb0.permute(1, 2, 0)
            emb1 = emb1.permute(1, 2, 0)
            emb2 = emb2.permute(1, 2, 0)
            emb3 = emb3.permute(1, 2, 0)

            for x, image in enumerate(torch.stack([emb0, emb1, emb2, emb3], dim=1)):
                embedded = image.cpu()
                with open(f"embeddings/embedding_{(i*16) + x}.pkl", "wb") as f:
                    pickle.dump(embedded, f)

    embedded_ds = EmbeddedDataset()
    loader = DataLoader(embedded_ds, batch_size=64,
                        shuffle=False, num_workers=4)

    cls = MGMTClassifier()
    with open("embedder.pkl", "rb") as f:
        cls.load_state_dict(pickle.load(f))
        cls = cls.to("cuda:0")

    pred_classes = pd.DataFrame(columns=["MGMT_probability", "MGMT_value"])

    cls.eval()
    with torch.no_grad():
        for n, batch in enumerate(loader):
            x, _ = batch
            data = x.to("cuda:0")
            embedded = cls(data)

            pred_cls = torch.argmax(embedded, dim=1)
            probs = embedded[:, pred_cls]

            for i, (prob, val) in enumerate(zip(probs.cpu(), pred_cls.cpu())):
                pred_classes.loc[(n * 64) + i] = [prob, val]

    pred_cls.index.name = "ID"
    pred_cls.index = pred_cls.map(lambda x: f"test_{x}")
    pred_cls.to_csv("submission.csv", index=True)
