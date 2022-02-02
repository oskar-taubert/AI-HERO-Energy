import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomLoadDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, device=None, normalize=True, metadata=False):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon
        self.metadata = metadata

        # Load Data from csv to Pandas Dataframe
        raw_data = pd.read_csv(data_file, delimiter=',')
        load_data = raw_data['Load [MWh]'].to_numpy()
        self.dataset = torch.Tensor(load_data)
        if self.metadata:
            self.metadata = raw_data['Time [s]']

        # Normalize Data to [0,1]
        if normalize is True:
            self.data_min = torch.min(self.dataset)
            self.data_max = torch.max(self.dataset)
            self.dataset = (self.dataset - self.data_min) / (self.data_max - self.data_min)

        self.dataset = self.dataset.to(device)

    def __len__(self):
        return int(self.dataset.shape[0] - self.historic_window - self.forecast_horizon)

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        x = self.dataset[idx:idx+self.historic_window].unsqueeze(dim=1)
        if self.metadata:
            x = (x, self.metadata.iloc[idx:idx+self.historic_window])
        y = self.dataset[idx+self.historic_window: idx+self.historic_window + self.forecast_horizon].unsqueeze(dim=1)

        return x, y

    def revert_normalization(self, data):
        return data * (self.data_max - self.data_min) + self.data_min
