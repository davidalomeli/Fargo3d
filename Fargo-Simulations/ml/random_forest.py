import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class SequenceDataGenerator:

    def __init__(self, data_loader, variables_calculator, dust_num, sequence_length):
        self.loader          = data_loader
        self.vars            = variables_calculator
        self.dust_num        = dust_num
        self.sequence_length = sequence_length

    def generate_sequences(self, orbit_list):
        X_seq, Y_seq = [], []
        for orbit in orbit_list:
            gas_dens         = self.loader.load_gasdens(orbit).flatten()
            vr, vphi         = self.loader.load_gasvel(orbit)
            gas_pressure     = self.vars.calculate_gas_pressure(orbit).flatten()
            gas_pressure_grad = self.vars.calculate_pressure_gradient(orbit).flatten()

            features = np.stack([gas_dens, vr.flatten(), vphi.flatten(), gas_pressure, gas_pressure_grad], axis=-1)
            X_seq.append(features)

            dust_dens        = self.loader.load_dustdens(orbit, self.dust_num).flatten()
            dust_vr, dust_vphi = self.loader.load_dustvel(orbit, self.dust_num)
            Y_seq.append(np.stack([dust_dens, dust_vr.flatten(), dust_vphi.flatten()], axis=-1))

        X_seq = np.array(X_seq).reshape(-1, np.array(X_seq).shape[-1])
        Y_seq = np.array(Y_seq).reshape(-1, np.array(Y_seq).shape[-1])
        return X_seq, Y_seq


class ML_Test:

    def __init__(self, data_loader, variables_calculator):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator

    def prepare_training_data(self, orbit, dust_num):
        gas_dens   = self.data_loader.load_gasdens(orbit)
        vr, vphi   = self.data_loader.load_gasvel(orbit)
        dust_dens  = self.data_loader.load_dustdens(orbit, dust_num)
        dust_vr, dust_vphi = self.data_loader.load_dustvel(orbit, dust_num)
        gas_pres   = self.variables_calculator.calculate_gas_pressure(orbit)
        gas_grad   = self.variables_calculator.calculate_pressure_gradient(orbit)

        features         = np.zeros((gas_dens.size, 5))
        features[:, 0]   = gas_dens.flatten()
        features[:, 1]   = vr.flatten()
        features[:, 2]   = vphi.flatten()
        features[:, 3]   = gas_pres.flatten()
        features[:, 4]   = gas_grad.flatten()

        targets          = np.zeros((gas_dens.size, 3))
        targets[:, 0]    = dust_dens.flatten()
        targets[:, 1]    = dust_vr.flatten()
        targets[:, 2]    = dust_vphi.flatten()
        return features, targets

    def train_model(self, features, targets):
        indices = np.arange(len(features))
        x_tr, x_te, y_tr, y_te, _, idx_te = train_test_split(
            features, targets, indices, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_tr, y_tr)
        return x_te, y_te, model, None, idx_te

    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        mse    = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return y_pred, y_test, mse


class ML_Predictor:

    def __init__(self, data_loader, variables_calculator, orbits, dust_num):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.sequence             = SequenceDataGenerator(
            data_loader, variables_calculator, dust_num, len(orbits)
        )
        self.dust_num = dust_num

    def training_data(self, orbits):
        return self.sequence.generate_sequences(orbits)

    def train_model(self, orbits):
        x_train, y_train = self.training_data(orbits)
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        return model

    def predict_dust(self, model, orbit):
        gasdens          = self.data_loader.load_gasdens(orbit).flatten()
        vr, vphi         = self.data_loader.load_gasvel(orbit)
        gas_pres         = self.variables_calculator.calculate_gas_pressure(orbit).flatten()
        gas_grad         = self.variables_calculator.calculate_pressure_gradient(orbit).flatten()
        x_input          = np.stack([gasdens, vr.flatten(), vphi.flatten(), gas_pres, gas_grad], axis=-1)
        return model.predict(x_input)

    def get_true_values(self, orbit):
        dust_dens        = self.data_loader.load_dustdens(orbit, self.dust_num).flatten()
        dust_vr, dust_vphi = self.data_loader.load_dustvel(orbit, self.dust_num)
        return np.stack([dust_dens, dust_vr.flatten(), dust_vphi.flatten()], axis=-1)
