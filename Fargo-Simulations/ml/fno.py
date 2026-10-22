import numpy as np
import torch
import torch.nn as nn
import torch.fft
from sklearn.preprocessing import MinMaxScaler


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_r, modes_phi):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_r      = modes_r
        self.modes_phi    = modes_phi
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_r, modes_phi, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, Nr, Nphi = x.shape
        x_ft  = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, Nr, Nphi // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_r, :self.modes_phi] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_r, :self.modes_phi],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=(Nr, Nphi))


class FNO2d(nn.Module):
    def __init__(self, in_channels=5, width=64, modes_r=16, modes_phi=16):
        super().__init__()
        self.fc0   = nn.Linear(in_channels, width)
        self.conv1 = SpectralConv2d(width, width, modes_r, modes_phi)
        self.conv2 = SpectralConv2d(width, width, modes_r, modes_phi)
        self.w1    = nn.Conv2d(width, width, 1)
        self.w2    = nn.Conv2d(width, width, 1)
        self.fc1   = nn.Linear(width, 128)
        self.fc2   = nn.Linear(128, 1)

    def forward(self, x):
        x  = self.fc0(x).permute(0, 3, 1, 2)
        x  = self.conv2(self.conv1(x) + self.w1(x)) + self.w2(self.conv1(x) + self.w1(x))
        x  = x.permute(0, 2, 3, 1)
        return self.fc2(torch.relu(self.fc1(x)))


class FNO_Predictor_Test:

    def __init__(self, data_loader, variables_calculator, orbits, dust_num, gas_threshold=1e-9):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.orbits               = orbits
        self.dust_num             = dust_num
        self.gas_threshold        = gas_threshold
        self.scalers = {k: MinMaxScaler() for k in ["gasdens", "vr", "vphi", "pressure", "grad", "dust"]}

    def safe_log(self, x):
        return np.log10(np.clip(x, 1e-20, None))

    def training_data(self, orbits):
        features, targets = [], []
        for orbit in orbits:
            gasdens  = self.safe_log(self.data_loader.load_gasdens(orbit))
            vr, vphi = self.data_loader.load_gasvel(orbit)
            gas_pres = self.safe_log(self.variables_calculator.calculate_gas_pressure(orbit))
            gas_grad = self.variables_calculator.calculate_pressure_gradient(orbit)
            dust_dens = self.safe_log(self.data_loader.load_dustdens(orbit, self.dust_num))
            features.append([gasdens, vr, vphi, gas_pres, gas_grad])
            targets.append(dust_dens)

        features = np.moveaxis(np.array(features, dtype=np.float32), 1, -1)
        targets  = np.array(targets, dtype=np.float32)

        for i, key in enumerate(["gasdens", "vr", "vphi", "pressure", "grad"]):
            flat = features[..., i].reshape(-1, 1)
            self.scalers[key].fit(flat)
            features[..., i] = self.scalers[key].transform(flat).reshape(features[..., i].shape)

        flat_t  = targets.reshape(-1, 1)
        self.scalers["dust"].fit(flat_t)
        targets = self.scalers["dust"].transform(flat_t).reshape(targets.shape)[..., None]

        return features.astype(np.float32), targets.astype(np.float32)

    def build_model(self):
        return FNO2d(in_channels=5, width=64, modes_r=16, modes_phi=16)

    def train_model(self, orbits, epochs=10):
        x_train, y_train = self.training_data(orbits)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = self.build_model().to(device)
        x_t    = torch.tensor(x_train).to(device)
        y_t    = torch.tensor(y_train).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            loss = loss_fn(model(x_t), y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")

        return model

    def predict_dust(self, model, orbit):
        device   = next(model.parameters()).device
        scaler   = MinMaxScaler()
        gasdens  = self.safe_log(self.data_loader.load_gasdens(orbit))
        vr, vphi = self.data_loader.load_gasvel(orbit)
        gas_pres = self.safe_log(self.variables_calculator.calculate_gas_pressure(orbit))
        gas_grad = self.variables_calculator.calculate_pressure_gradient(orbit)

        fields = [gasdens, vr, vphi, gas_pres, gas_grad]
        keys   = ["gasdens", "vr", "vphi", "pressure", "grad"]
        scaled = [self.scalers[k].transform(f.reshape(-1,1)).reshape(f.shape) for k, f in zip(keys, fields)]
        x_input = torch.tensor(np.expand_dims(np.stack(scaled, axis=-1), 0), dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(x_input)[0, ..., 0].cpu().numpy()

        return self.scalers["dust"].inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)

    def get_true_values(self, orbit):
        return self.data_loader.load_dustdens(orbit, self.dust_num)
