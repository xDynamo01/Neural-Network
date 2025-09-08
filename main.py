import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# -----------------------------
# Dense Layer
# -----------------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradiente em relação aos pesos e bias
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradiente para a camada anterior
        self.dinputs = np.dot(dvalues, self.weights.T)

# -----------------------------
# ReLU
# -----------------------------
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# -----------------------------
# Softmax
# -----------------------------
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # backward normal não é eficiente para cross-entropy
        # vamos deixar só placeholder
        self.dinputs = dvalues.copy()

# -----------------------------
# Loss base
# -----------------------------
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

# Categorical Crossentropy
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = (dvalues - y_true) / samples

# -----------------------------
# Otimizador (SGD)
# -----------------------------
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# -----------------------------
# Teste - spiral dataset
# -----------------------------
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.1)

# Treinamento
for epoch in range(1001):
    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Loss
    loss = loss_function.calculate(activation2.output, y)

    # Acurácia
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

    # Backward
    loss_function.backward(activation2.output, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Atualização
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


# Depois do treinamento
import pickle

model = {
    'dense1': {'weights': dense1.weights, 'biases': dense1.biases},
    'dense2': {'weights': dense2.weights, 'biases': dense2.biases}
}

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo salvo em model.pkl")
# Criar rede "vazia" (mesma estrutura)
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

# Carregar pesos salvos
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

dense1.weights = model['dense1']['weights']
dense1.biases = model['dense1']['biases']
dense2.weights = model['dense2']['weights']
dense2.biases = model['dense2']['biases']

print("Modelo carregado com sucesso!")

# Fazer previsão com o modelo carregado
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

predictions = np.argmax(activation2.output, axis=1)
print("Predições:", predictions[:10])

# -----------------------------
# Visualização da fronteira de decisão
# -----------------------------
import matplotlib.pyplot as plt

# Gerar uma grade de pontos
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Passar a grade pela rede treinada
grid_points = np.c_[xx.ravel(), yy.ravel()]
dense1.forward(grid_points)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Predições da rede
Z = np.argmax(activation2.output, axis=1)
Z = Z.reshape(xx.shape)

# Plotar as regiões
plt.contourf(xx, yy, Z, alpha=0.4, cmap="brg")

# Plotar os dados originais
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg", edgecolors='k')
plt.title("Fronteira de decisão da rede")
plt.show()


#Grafico

import matplotlib.pyplot as plt

loss_history = []
acc_history = []

for epoch in range(1001):
    # Forward
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Loss
    loss = loss_function.calculate(activation2.output, y)

    # Accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Guardar histórico
    loss_history.append(loss)
    acc_history.append(accuracy)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

    # Backward
    loss_function.backward(activation2.output, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Atualização
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# -----------------------------
# Gráficos
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history, label="Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Evolução da Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(acc_history, label="Accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Evolução da Acurácia")
plt.legend()

plt.show()
