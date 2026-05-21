from torch.utils.data import DataLoader
from dataset import FloodSTGCNDataset

dataset = FloodSTGCNDataset(
    x_path=r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\features\X_dynamic.npz",
    target_path=r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\labels\p97\fold_2_train2016_2021_val2022_test2023\targets.npz",
    timestamp_path=r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\features\timestamps.csv",
    split="train",
)

loader = DataLoader(dataset, batch_size=4)
x, y, mask = next(iter(loader))

print("X shape:", x.shape)       # 기대 결과: [4, 12, 329, 8]
print("Y shape:", y.shape)       # 기대 결과: [4, 329]
print("Mask shape:", mask.shape) # 기대 결과: [4, 329]