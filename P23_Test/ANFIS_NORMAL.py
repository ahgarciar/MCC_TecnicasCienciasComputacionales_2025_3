import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class ANFIS:
    # 1. LA INICIALIZACION CORRECTA (CON SIGMA INTELIGENTE)
    def __init__(self, n_inputs, n_rules, x_range):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.mf_params = np.zeros((self.n_rules, self.n_inputs, 2))

        centers = np.linspace(x_range[0], x_range[1], self.n_rules)
        for i in range(self.n_inputs):
            self.mf_params[:, i, 0] = centers

        if self.n_rules > 1:
            sigma = (x_range[1] - x_range[0]) / (2 * (self.n_rules - 1))
        else:
            sigma = (x_range[1] - x_range[0]) / 2
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
        rule_outputs = np.dot(inputs_with_bias, self.rule_params.T)
        final_output = np.sum(normalized_w * rule_outputs, axis=1)
        intermediates = {"mu": mu, "w": w, "normalized_w": normalized_w, "rule_outputs": rule_outputs,
                         "inputs_with_bias": inputs_with_bias}
        return final_output, intermediates

    # 2. EL METODO TRAIN CON EL SIGNO CORREGIDO
    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred, intermediates = self.forward_pass(X_train)
            error = y_train.flatten() - y_pred.flatten()

            # Actualizar parametros del consecuente
            error_reshaped = error.reshape(-1, 1)
            grad_rule = np.dot((-error_reshaped * intermediates['normalized_w']).T, intermediates['inputs_with_bias'])
            self.rule_params -= learning_rate * grad_rule

            # Actualizar parametros de la premisa
            # CORRECCION CLAVE: Se agrega un signo de menos a 'error_reshaped'
            # para asegurar que sea descenso de gradiente.
            common_term = -error_reshaped * (intermediates['rule_outputs'] - y_pred.reshape(-1, 1)) / (
                        np.sum(intermediates['w'], axis=1, keepdims=True) + 1e-10)

            grad_mf = np.zeros_like(self.mf_params)
            for i in range(self.n_rules):
                for j in range(self.n_inputs):
                    mu_ij, w_i_stable = intermediates['mu'][:, i, j], intermediates['w'][:, i] + 1e-10
                    dw_dmu = w_i_stable / (mu_ij + 1e-10)
                    c, sigma = self.mf_params[i, j]

                    dmu_dc = mu_ij * (X_train[:, j] - c) / (sigma ** 2 + 1e-10)
                    grad_mf[i, j, 0] = np.sum(common_term[:, i] * dw_dmu * dmu_dc)

                    dmu_dsigma = mu_ij * ((X_train[:, j] - c) ** 2) / (sigma ** 3 + 1e-10)
                    grad_mf[i, j, 1] = np.sum(common_term[:, i] * dw_dmu * dmu_dsigma)

            self.mf_params -= learning_rate * grad_mf

            if epoch % 500 == 0:
                print("Epoch "+str(epoch)+" Error: "+ str(np.mean(np.abs(error))) +"")

    def predict(self, inputs):
        if inputs.ndim == 1: inputs = inputs.reshape(-1, 1)
        output, _ = self.forward_pass(inputs)
        return output


# --- Generacion y escalado de datos (sin cambios) ---
x = np.linspace(-10, 10, 1001)
y = np.sinc(x) * np.sin(x)
input_data, output_data = x.reshape(-1, 1), y.reshape(-1, 1)

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
X_train_scaled, y_train_scaled = scaler_x.fit_transform(X_train), scaler_y.fit_transform(y_train)
X_test_scaled = scaler_x.transform(X_test)

# --- Entrenamiento y Prediccion ---
n_rules = 40
n_inputs = X_train_scaled.shape[1]

anfis_model = ANFIS(n_inputs=n_inputs, n_rules=n_rules, x_range=[0, 1])

# Usamos una tasa de aprendizaje moderada
anfis_model.train(X_train_scaled, y_train_scaled, epochs=5000, learning_rate=0.005)

y_pred_scaled = anfis_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# --- Visualizacion de Resultados ---
plt.figure(figsize=(12, 7))
plt.plot(input_data, output_data, color='gray', linestyle='--', label='Funcion Original')
plt.scatter(X_train, y_train, color='blue', s=10, label='Datos de Entrenamiento', alpha=0.2)
sort_axis = np.argsort(X_test.flatten())
plt.plot(X_test[sort_axis], y_pred[sort_axis], color='red', linewidth=3, label='Prediccion ANFIS (Corregida)')
plt.title('Aproximacion de f(x) = sinc(x) * sin(x) con ANFIS')
plt.xlabel('x'), plt.ylabel('f(x)'), plt.legend(), plt.grid(True)
plt.show()
