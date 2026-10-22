import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


class CNN_Predictor:

    def __init__(self, data_loader, variables_calculator, orbits, dust_num, gas_threshold=1e-9):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.n_orbits             = len(orbits)
        self.dust_num             = dust_num
        self.gas_threshold        = gas_threshold

    def training_data(self, orbits):
        features, targets, masks = [], [], []
        max_dust = -np.inf
        scaler   = MinMaxScaler()

        for orbit in orbits:
            gas_dens  = self.data_loader.load_gasdens(orbit)
            vr, vphi  = self.data_loader.load_gasvel(orbit)
            gas_pres  = self.variables_calculator.calculate_gas_pressure(orbit)
            gas_grad  = self.variables_calculator.calculate_pressure_gradient(orbit)

            mask     = gas_dens > self.gas_threshold
            gas_dens = np.log10(gas_dens)
            gas_dens = np.where(mask, gas_dens, 0.0)
            vr       = np.where(mask, vr,       0.0)
            vphi     = np.where(mask, vphi,     0.0)
            gas_pres = np.where(mask, gas_pres, 0.0)
            gas_grad = np.where(mask, gas_grad, 0.0)

            fmap = np.stack([gas_dens, vr, vphi, gas_pres, gas_grad], axis=-1)
            for i in range(fmap.shape[-1]):
                fmap[..., i] = scaler.fit_transform(fmap[..., i].reshape(-1, 1)).reshape(fmap[..., i].shape)
            features.append(fmap)

            dust_dens        = self.data_loader.load_dustdens(orbit, self.dust_num)
            dust_vr, dust_vphi = self.data_loader.load_dustvel(orbit, self.dust_num)
            max_dust         = max(max_dust, np.max(dust_dens))
            tmap = np.stack([
                np.where(mask, dust_dens,  0.0),
                np.where(mask, dust_vr,    0.0),
                np.where(mask, dust_vphi,  0.0),
            ], axis=-1)
            targets.append(tmap)
            masks.append(mask.astype(np.float32))

        return (
            np.array(features, dtype=np.float32),
            np.array(targets,  dtype=np.float32),
            np.array(masks,    dtype=np.float32),
            max_dust,
        )

    def build_model(self, input_shape):
        n = 3
        model = models.Sequential([
            layers.Conv3D(32,  (n, 3, 3), activation="relu", padding="same", input_shape=input_shape),
            layers.Conv3D(64,  (n, 3, 3), activation="relu", padding="same"),
            layers.Conv3D(128, (n, 3, 3), activation="relu", padding="same"),
            layers.Conv3D(3,   (1, 5, 5), activation="relu", padding="same"),
        ])
        model.compile(optimizer="adam", loss=self.masked_mse)
        return model

    def masked_mse(self, y_true, y_pred):
        mask        = tf.cast(y_true[..., 0] > 0, tf.float32)
        masked_diff = (y_true - y_pred) ** 2 * tf.expand_dims(mask, axis=-1)
        return tf.reduce_sum(masked_diff) / (tf.reduce_sum(mask) + 1e-6)

    def train_model(self, orbits, epochs):
        x, y, masks, max_density = self.training_data(orbits)
        if x.ndim == 4:
            x = x[np.newaxis]; y = y[np.newaxis]
        model = self.build_model(x.shape[1:])
        model.fit(x, y, epochs=epochs, batch_size=16)
        return model, max_density

    def predict_dust(self, model, orbits, max_dust):
        scaler, sequence = MinMaxScaler(), []
        for orbit in orbits:
            gasdens   = self.data_loader.load_gasdens(orbit)
            vr, vphi  = self.data_loader.load_gasvel(orbit)
            gas_pres  = self.variables_calculator.calculate_gas_pressure(orbit)
            gas_grad  = self.variables_calculator.calculate_pressure_gradient(orbit)
            mask      = gasdens > self.gas_threshold
            gasdens   = np.where(mask, np.log10(gasdens), 0.0)

            x = np.stack([gasdens, np.where(mask, vr, 0.0), np.where(mask, vphi, 0.0),
                          np.where(mask, gas_pres, 0.0), np.where(mask, gas_grad, 0.0)], axis=-1)
            for i in range(x.shape[-1]):
                x[..., i] = scaler.fit_transform(x[..., i].reshape(-1, 1)).reshape(x[..., i].shape)
            sequence.append(x)

        x_input = np.expand_dims(np.array(sequence), axis=0)
        pred    = model.predict(x_input)[0]
        pred[..., 0] *= max_dust
        return pred

    def get_true_values(self, orbit):
        dust_dens        = self.data_loader.load_dustdens(orbit, self.dust_num)
        dust_vr, dust_vphi = self.data_loader.load_dustvel(orbit, self.dust_num)
        return np.stack([dust_dens, dust_vr, dust_vphi], axis=-1)
