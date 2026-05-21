from torch.utils.data import DataLoader

from dataset import FloodSTGCNDataset


dataset = FloodSTGCNDataset(
    x_path=r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\features\X_dynamic.npz",

    target_path=r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\labels\p97\fold_2_train2016_2021_val2022_test2023\targets.npz",

    split="train",
)

loader = DataLoader(dataset, batch_size=4)

x, y, mask = next(iter(loader))

print(x.shape)
print(y.shape)
print(mask.shape)