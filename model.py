import torch
from torch import nn


class LoadForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layer: int = 1, dropout: float = 0, batch_first: bool = True, device: torch.device = None):
        super(LoadForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, dropout=dropout, batch_first=batch_first,
                            device=self.device)
        self.fully_connected = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(self, input_sequence, hidden):
        output, hidden = self.lstm(input_sequence, hidden)
        output = self.fully_connected(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device)

        return hidden_state, cell_state


class NaiveModel(nn.Module):
    def __init__(self, training_data):
        self.holidays = holidays.CountryHoliday('DE', prov=None, state='NI')
        # TODO learn weights and values here
        self.weeks = torch.empty((53, 24,), dtype=float)
        self.seasons = torch.empty((7, 24,), dtype=float)
        self.holidays = torch.empty((2, 24,), dtype=float)
        w1 = None
        w2 = None
        w3 = None

    def forward(self, metadata):
        d = datetime.date(*metadata.split('-'))
        _, week, day = d.isocalendar()
        holiday = metadata in self.holidays

        return self.w1 * self.weeks[day] + self.w2 * self.seasons[weeks] + self.w3 * sekf,holiday[int(holiday)]


class SophisticatedModel(LightningModule):

    def __init__(self, naive_model):
        self.naive_model = naive_model
        # self.transformer = nn.Transformer(d_model = 64,
        #         n_head = 4,
        #         num_encoder_layers = 2,
        #         num_decoder_layers = 6,
        #         dim_feedworward = 128,
        #         dropout=0.1,)


    def forward(self, x):
        return self.naive_model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)

        lossval = self.criterion(prediction, y)
        return lossval
