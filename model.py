import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn.parameter import Parameter
import datetime


class LoadForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layer: int = 1, dropout: float = 0, batch_first: bool = True, device: torch.device = None, use_finaldense=True):
        super(LoadForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.device = device
        self.use_finaldense = use_finaldense

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, dropout=dropout, batch_first=batch_first,
                            device=self.device)
        self.fully_connected = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(self, input_sequence, hidden):
        output, hidden = self.lstm(input_sequence, hidden)
        if self.use_finaldense:
            output = self.fully_connected(output)
        return output, hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = self.device
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)

        return hidden_state, cell_state


class NaiveModel(nn.Module):
    def __init__(self, seasonal_delta=torch.tensor([0.], dtype=torch.float),
        cosmic_slope=torch.tensor(0., dtype=torch.float),
        cosmic_intersection=torch.tensor(0., dtype=torch.float),
        device=None):
        super(NaiveModel, self).__init__()
        self.device = device
        # NOTE seasonal_delta: 0.
        # NOTE seasonal_delta: tensor of shape [53]
        # NOTE seasonal_delta: tensor of shape [53, 24]
        # NOTE seasonal_delta: tensor of shape [53, 7, 24]
        if len(seasonal_delta.size()) == 1:
            if seasonal_delta.size(0) == 1:
                seasonal_delta = seasonal_delta.expand(53)
            assert seasonal_delta.size(0) == 53
            seasonal_delta = seasonal_delta.unsqueeze(-1)
            seasonal_delta.expand(53, 24)
        if len(seasonal_delta.size()) == 2:
            seasonal_delta = seasonal_delta.unsqueeze(-2)
            seasonal_delta = seasonal_delta.expand(53, 7, 24)
        if len(seasonal_delta.size()) != 3:
            raise ValueError()

        # TODO these are not trainable parameters
        self.seasonal_delta = Parameter(seasonal_delta.clone().detach(), requires_grad=False)
        # self.holidays = holidays.CountryHoliday('DE', prov=None, state='NI')
        self.cosmic_intersection = Parameter(cosmic_intersection.clone(), requires_grad=False)
        self.cosmic_slope = Parameter(cosmic_slope.clone().detach(), requires_grad=False)
        self.year0 = 2015

    def forward(self, x):
        loaddata, metadata = x
        assert loaddata.size() == (len(metadata), len(metadata[0]), 1)
        # NOTE metadata is an iterable of batch_size lists of strings [[YYYY-MM-DD, YYYY-MM-DD, ...], ...] containing dates of the source sequence / historic window
        ds = [[datetime.date(*[int(x) for x in (s.split()[0].split('-'))]) for s in sample] for sample in metadata]
        assert loaddata.size() == (len(ds), len(ds[0]), 1)
        isods = [[d.isocalendar() for d in sample] for sample in ds] # [(year, week, day), ...]
        weeks = [[d[1] for d in sample] for sample in isods]
        assert loaddata.size() == (len(weeks), len(weeks[0]), 1)
        # holidays = [[int(m in self.holidays) for m in l] for l in metadata]
        years = [[d[0] for d in sample] for sample in isods]
        weekdays = [[d[2] for d in sample] for sample in isods]
        hours = [[int(s.split()[1].split(':')[0]) for s in sample] for sample in metadata]

        # out = torch.tensor([self.seasonal_delta[w] for w in weeks], dtype=torch.float, device=loaddata.device)
        batch_size = loaddata.size(0)
        out = torch.empty((batch_size, 7*24, 1), device=self.seasonal_delta.device)
        for i in range(batch_size):
            for j in range(7*24):
                out[i, j, 0] = self.seasonal_delta[weeks[i][j]-1, weekdays[i][j]-1, hours[i][j]]
                out[i, j, 0] += self.cosmic_intersection + self.cosmic_slope * (years[i][j] - self.year0)


        # NOTE returns torch.Tensor of shape [batch_size, 7 * 24, 1]
        return out


class SophisticatedModel(LightningModule):

    def __init__(self, naive_model, encoder, decoder, train_encoder=True, learning_rate=1e-3):
        super().__init__()
        self.naive_model = naive_model
        self.encoder = encoder
        self.decoder = decoder
        self.train_encoder = train_encoder
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # NOTE input: load data tensor in shape
        loaddata, metadata = x
        batch_size = loaddata.size(0)
        (h, c) = self.encoder.init_hidden(batch_size, device=loaddata.device)
        if self.train_encoder:
            encoder_out, (h, c) = self.encoder(loaddata, (h, c))
        else:
            with torch.no_grad():
                encoder_out, (h, c) = self.encoder(loaddata, (h, c))

        with torch.no_grad():
            naive_out = self.naive_model(x)
        decoder_out, _ = self.decoder(naive_out, (h, c))

        return loaddata + decoder_out

    def training_step(self, batch, batch_idx):
        x, y = batch  # ([batch_size, 24*7, 1], list of 24*7 strings), [batch_size, 24*7, 1]
        y = y

        prediction = self(x)

        lossval = self.criterion(prediction, y)
        return lossval

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ([batch_size, 24*7, 1], list of 24*7 strings), [batch_size, 24*7, 1]
        y = y

        prediction = self(x)

        lossval = self.criterion(prediction, y)
        return lossval

    def test_step(self, batch, batch_idx):
        raise

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
