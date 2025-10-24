import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class ANFIS:
    def __init__(self, n_inputs, n_rules, x_range):
        self.n_inputs, self.n_rules = n_inputs, n_rules
        self.mf_params = np.zeros((self.n_rules, self.n_inputs, 2))

        # CORRECCIÓN PARA EL ValueError
        # Reshape de 'centers' para que coincida con la forma de destino (n_rules, 1)
        centers = np.linspace(x_range[0], x_range[1], self.n_rules).reshape(self.n_rules, -1)

        # El bucle es útil si n_inputs > 1. Lo mantenemos por generalidad.
        for i in range(self.n_inputs):
            self.mf_params[:, i, 0] = centers.flatten()

        if n_rules > 1:
            sigma = (x_range[1] - x_range[0]) / (1.5 * (self.n_rules - 1))
        else:
            sigma = (x_range[1] - x_range[0])
        self.mf_params[:, :, 1] = sigma

        self.rule_params = np.random.rand(self.n_rules, self.n_inputs + 1)

    def gaussian_mf(self, x, c, sigma):
        return np.exp(-((x - c) ** 2) / (2 * (sigma ** 2 + 1e-10)))

    def forward_pass(self, inputs):
        if inputs.ndim == 1: inputs = inputs.reshape(-1, 1)
        mu = np.zeros((inputs.shape[0], self.n_rules, self.n_inputs))
        for i in range(self.n_rules):
            for j in range(self.n_inputs):
                c, sigma = self.mf_params[i, j]
                mu[:, i, j] = self.gaussian_mf(inputs[:, j], c, sigma)
        w = np.prod(mu, axis=2)
        sum_w = np.sum(w, axis=1, keepdims=True) + 1e-10
        normalized_w = w / sum_w
        inputs_with_bias = np.hstack([inputs, np.ones((inputs.shape[0], 1))])
        # Devolvemos 'mu' porque es necesario para el entrenamiento
        return normalized_w, inputs_with_bias, w, mu

    def train_hybrid(self, X_train, y_train, epochs, learning_rate=0.01):
        y_train_flat = y_train.flatten()
        for epoch in range(epochs):
            # --- PASE HACIA ADELANTE ---
            normalized_w, inputs_with_bias, w, mu = self.forward_pass(X_train)

            # --- PARTE 1: MÍNIMOS CUADRADOS (LSE) para Parámetros de Consecuente ---
            A = np.zeros((X_train.shape[0], self.n_rules * (self.n_inputs + 1)))
            for i in range(self.n_rules):
                A[:, i * (self.n_inputs + 1):(i + 1) * (self.n_inputs + 1)] = normalized_w[:, i].reshape(-1,
                                                                                                         1) * inputs_with_bias

            try:
                consequent_params, _, _, _ = np.linalg.lstsq(A, y_train_flat, rcond=None)
                self.rule_params = consequent_params.reshape(self.n_rules, self.n_inputs + 1)
            except np.linalg.LinAlgError:
                pass  # Si LSE falla, omitimos esta actualización en esta época

            # --- PARTE 2: DESCENSO DE GRADIENTE (GD) para Parámetros de Premisa (LÓGICA CORREGIDA) ---
            rule_outputs = np.dot(inputs_with_bias, self.rule_params.T)
            y_pred = np.sum(normalized_w * rule_outputs, axis=1)
            error = y_train_flat - y_pred
            error_reshaped = error.reshape(-1, 1)

            s = np.sum(rule_outputs * normalized_w, axis=1, keepdims=True)
            delta = -error_reshaped * (rule_outputs - s) / (np.sum(w, axis=1, keepdims=True) + 1e-10)

            grad_mf = np.zeros_like(self.mf_params)
            for i in range(self.n_rules):
                for j in range(self.n_inputs):
                    # Esta es la forma correcta de calcular la derivada parcial
                    w_div_mu = w[:, i].reshape(-1, 1) / (mu[:, i, j].reshape(-1, 1) + 1e-10)

                    dmu_dc = mu[:, i, j].reshape(-1, 1) * (X_train[:, j].reshape(-1, 1) - self.mf_params[i, j, 0]) / (
                                self.mf_params[i, j, 1] ** 2 + 1e-10)
                    dmu_dsigma = mu[:, i, j].reshape(-1, 1) * (
                                (X_train[:, j].reshape(-1, 1) - self.mf_params[i, j, 0]) ** 2) / (
                                             self.mf_params[i, j, 1] ** 3 + 1e-10)

                    grad_mf[i, j, 0] = np.sum(delta[:, i].reshape(-1, 1) * w_div_mu * dmu_dc)
                    grad_mf[i, j, 1] = np.sum(delta[:, i].reshape(-1, 1) * w_div_mu * dmu_dsigma)

            self.mf_params -= learning_rate * grad_mf

            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

    def predict(self, inputs):
        normalized_w, inputs_with_bias, _, _ = self.forward_pass(inputs)
        rule_outputs = np.dot(inputs_with_bias, self.rule_params.T)
        return np.sum(normalized_w * rule_outputs, axis=1)


# --- Generación y escalado de datos (sin cambios) ---
x = np.linspace(-10, 10, 1001)
y = np.sinc(x) * np.sin(x)
input_data, output_data = x.reshape(-1, 1), y.reshape(-1, 1)

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
X_train_scaled, y_train_scaled = scaler_x.fit_transform(X_train), scaler_y.fit_transform(y_train)

# --- Entrenamiento y Predicción ---
n_rules = 40
n_inputs = X_train_scaled.shape[1]
anfis_model = ANFIS(n_inputs=n_inputs, n_rules=n_rules, x_range=[0, 1])

print("Entrenando el modelo con el algoritmo híbrido...")
anfis_model.train_hybrid(X_train_scaled, y_train_scaled, epochs=100, learning_rate=0.01)
print("Entrenamiento finalizado.")

# --- Predicción y Visualización (Metodología del director) ---
input_data_scaled = scaler_x.transform(input_data)
y_pred_scaled_full = anfis_model.predict(input_data_scaled)
y_pred_full = scaler_y.inverse_transform(y_pred_scaled_full.reshape(-1, 1))

plt.figure(figsize=(12, 7))
plt.plot(input_data, output_data, color='gray', linestyle='--', label='Función Original')
plt.scatter(X_train, y_train, color='blue', s=10, label='Datos de Entrenamiento', alpha=0.3)
plt.plot(input_data, y_pred_full, color='red', linewidth=3, label='Predicción ANFIS Híbrido')
plt.title('Aproximación de f(x) = sinc(x) · sin(x) con ANFIS Híbrido')
plt.xlabel('x'), plt.ylabel('f(x)'), plt.legend(), plt.grid(True)
plt.show()