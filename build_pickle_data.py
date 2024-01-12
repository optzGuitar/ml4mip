import pickle
from data.classification import ClassificationDataset
from concurrent.futures import ProcessPoolExecutor


if __name__ == "__main__":
    train_ds = ClassificationDataset(full_augment=False)

    def pickle_id(id: int):
        data = train_ds[id]
        with open(f"class_data/{id}.pkl", "wb") as f:
            pickle.dump(data, f)

    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(pickle_id, range(len(train_ds)))
