#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LoadForecaster
from dataset import CustomLoadDataset
from pytorch_lightning.callbacks import ModelCheckpoint


forecast_days = 7


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--num_epochs", type=int, default=30)
    # parser.add_argument("--save_dir", default=None, help="saves the model, if path is provided")
    parser.add_argument("--historic_window", type=int, default=7*24, help="input time steps in hours")
    parser.add_argument("--forecast_horizon", type=int, default=forecast_days*24, help="forecast time steps in hours")
    parser.add_argument("--hidden_size", type=int, default=48, help="size of the internal state")
    parser.add_argument("--decoder_hidden_size", type=int, default=48, help="size of the internal state")
    parser.add_argument("--encoder_depth", type=int, default=2, help="size of the internal state")
    parser.add_argument("--decoder_depth", type=int, default=2, help="size of the internal state")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--naive_file", type=str, default='seasonal_delta.pt')

    args = parser.parse_args()

    gpus = 4 if torch.cuda.is_available() else None

    # Forecast Parameters
    historic_window = args.historic_window
    forecast_horizon = args.forecast_horizon

    # Loading Data
    data_dir = args.data_dir
    if data_dir == '':
        data_dir = os.environ['AIHERO_PATH']
    train_set = CustomLoadDataset(
        os.path.join(data_dir, 'train.csv'),
        historic_window, forecast_horizon, device, metadata=True)
    valid_set = CustomLoadDataset(
        os.path.join(data_dir, 'valid.csv'),
        historic_window, forecast_horizon, device, metadata=True)

    # Create DataLoaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Configuring Model
    hidden_nodes = args.hidden_size
    decoder_hidden_nodes = args.decoder_hidden_size
    encoder_depth = args.encoder_depth
    decoder_depth = args.decoder_depth
    input_size = 1
    output_size = 1

    naive_file = args.naive_file
    if naive_file == '':
        # NOTE persistence model
        naive_params = torch.tensor([0.], dtype=torch.float)
    else:
        # TODO adapt to improved naive model
        naive_params  = torch.load(naive_file)

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    naive_model = NaiveModel(naive_params)
    encoder = LoadForecaster(input_size, hidden_nodes, output_size, num_layer=encoder_depth, use_finaldense=True)
    decoder = LoadForecaster(input_size, decoder_hidden_nodes, output_size, num_layer=decoder_depth, use_finaldense=True)
    model = SophisticatedModel(naive_model, encoder=encoder)

    criterion = nn.MSELoss()

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath='/hkfs/work/workspace/scratch/bh6321-E2/AI-HERO-Energy/', filename='submitted_model.pt')

    trainer = Trainer(gpus=gpus, callbacks=[checkpoint_callback])
    trainer.fit(mode, train_loader, val_loader)

if __name__ == '__main__':
    main()
