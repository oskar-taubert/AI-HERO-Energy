import torch
from torch import nn


class LoadForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layer: int = 1, dropout: float = 0, batch_first: bool = True, device: torch.device = None, use_finaldense=True):
        super(LoadForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, dropout=dropout, batch_first=batch_first,
                            device=self.device)
        self.fully_connected = nn.Linear(hidden_size, output_size, device=self.device)

    def forward(self, input_sequence, hidden):
        output, hidden = self.lstm(input_sequence, hidden)
        if self.use_finaldense:
            output = self.fully_connected(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device)

        return hidden_state, cell_state


class NaiveModel(nn.Module, seasonal_bias):
    def __init__(self, seasonal_bias=torch.tensor([0.], dtype=torch.float()), device=None):
        self.device = device
        # NOTE seasonal_bias: 0.
        # NOTE seasonal_bias: tensor of shape [53]
        # NOTE seasonal_bias: tensor of shape [53, 24]
        # NOTE seasonal_bias: tensor of shape [53, 7, 24]
        if len(seasonal_bias.size()) == 1:
            if seasonal_bias.size(0) == 1:
                seasonal_bias.expand(53)
            seasonal_bias.unsqueeze(-1)
            seasonal_bias.expand(53, 24)
        if len(seasonal_bias.size()) == 2:
            seasonal_bias.unsqueeze(-2)
            seasonal_bias.expand(53, 7, 24)
        if len(seasonal_bias.size()) != 3:
            raise ValueError()

        self.seasonal_bias = Parameter(torch.tensor(seasonal_bias), device=device)
        # self.holidays = holidays.CountryHoliday('DE', prov=None, state='NI')
        # self.cosmic_bias = torch.empty((24), dtype=torch.float)
        # self.cosmic_gradient = torch.empty((24), dtype=torch.float)

    def forward(self, x):
        loaddata, metadata = x
        # NOTE metadata is an iterable of batch_size lists of strings [[YYYY-MM-DD, YYYY-MM-DD, ...], ...] containing dates of the source sequence / historic window
        ds = [datetime.date(*s.split('-')) for s in metadata]
        isods = [d.isocalender() for d in ds] # [(year, week, day), ...]
        # years = [d[0] for d in isods]
        weeks = [d[1] for d in isods]
        # days  = [d[2] for d in isods]
        # holidays = [[int(m in self.holidays) for m in l] for l in metadata]

        out = torch.tensor([self.seasonal_bias[w] for w in weeks], dtype=torch.float)

        # NOTE returns torch.Tensor of shape [batch_size, 7 * 24, 1]
        return out


# TODO correct batch collation
class SophisticatedModel(LightningModule):

    def __init__(self, naive_model, encoder, decoder, train_encoder=False):
        self.naive_model = naive_model
        self.encoder = encoder
        self.decoder = decoder
        self.train_encoder = train_encoder

    def forward(self, x):
        # NOTE input: load data tensor in shape
        loaddata, metadata = x
        batch_size = x.size(1)
        (h, c) = self.encoder.init_hidden(batch_size)
        if self.train_encoder:
            encoder_out, (h, c) = self.encoder(x (h, c))
        else:
            with torch.no_grad():
                encoder_out, (h, c) = self.encoder(x (h, c))
        decoder_out = self.decoder(encoder_out, (h, c))

        return output, (h, c)

    def training_step(self, batch, batch_idx):
        x, y = batch  # ([batch_size, 24*7, 1], list of 24*7 strings), [batch_size, 24*7, 1]
        y = y - x[0]

        prediction, _ = self(x)

        lossval = self.criterion(prediction, y)
        return lossval

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ([batch_size, 24*7, 1], list of 24*7 strings), [batch_size, 24*7, 1]
        y = y - x[0]

        prediction, _ = self(x)

        lossval = self.criterion(prediction, y)
        return lossval

    def test_step(self, batch, batch_idx):
        raise

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer
