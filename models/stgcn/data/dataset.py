import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FloodSTGCNDataset(Dataset):
    def __init__(
        self,
        x_path,
        target_path,
        timestamp_path,
        seq_len=12,
        pred_horizon=1,
        split="train",
    ):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        # 1. 데이터 로드
        x_data = np.load(x_path)
        self.X = x_data["X_dynamic"].astype(np.float32)

        target_data = np.load(target_path)
        self.z = target_data["z"].astype(np.float32)
        self.z_mask = target_data["z_mask"].astype(np.bool_)

        # 2. 타임스탬프 로드 및 컬럼명 유연하게 대처
        df_time = pd.read_csv(timestamp_path)
        # 컬럼명이 'timestamp'든 'timestamps'든 첫 번째 컬럼이든 매핑되도록 처리
        time_col = [col for col in df_time.columns if 'time' in col.lower()]
        if time_col:
            timestamps = pd.to_datetime(df_time[time_col[0]])
        else:
            timestamps = pd.to_datetime(df_time.iloc[:, 0])

        # 3. 유효 인덱스 수집
        self.valid_indices = []
        T = self.X.shape[0]

        for t in range(seq_len - 1, T - pred_horizon):
            target_t = t + pred_horizon
            target_year = timestamps[target_t].year

            # Split 필터링
            if split == "train" and not (2016 <= target_year <= 2021):
                continue
            elif split == "val" and target_year != 2022:
                continue
            elif split == "test" and target_year != 2023:
                continue

            # [수정] z_mask.sum() == 0 조건을 제거합니다.
            # 평상시(마비 진입 노드가 없는 시점) 데이터도 모델이 안정적으로 공백을 예측하도록 학습해야 합니다.

            self.valid_indices.append(t)

        print(f"[{split}] valid samples found: {len(self.valid_indices)}")
        
        # 데이터가 아예 없을 경우를 위한 디버깅 경고성 방어 코드
        if len(self.valid_indices) == 0:
            print(f"⚠️ 경고: [{split}] 스플릿 조건에 맞는 데이터가 존재하지 않습니다.")
            print(f"전체 데이터 시점 개수 T: {T}, 데이터 내 실제 연도 범위: {timestamps.dt.year.min()} ~ {timestamps.dt.year.max()}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        x_start = t - self.seq_len + 1
        x_end = t + 1
        target_t = t + self.pred_horizon

        x = torch.from_numpy(self.X[x_start:x_end])
        y = torch.from_numpy(self.z[target_t])
        mask = torch.from_numpy(self.z_mask[target_t])

        return x, y, mask