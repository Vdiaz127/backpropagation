import numpy as np
import json

# Función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cargar dataset simulado (se puede reemplazar con datos reales)
def load_dataset():
    with open('maquinas.json', 'r') as file:
        data = json.load(file)
    
    X = np.array([[d['temperatura'], d['presion'], d['corriente']] for d in data])
    y = np.array([d['fallo'] for d in data]).reshape(-1, 1)
    
    return X, y

# Inicializar pesos y sesgos aleatoriamente
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.rand(input_size, hidden_size) - 0.5
    b1 = np.random.rand(1, hidden_size) - 0.5
    W2 = np.random.rand(hidden_size, output_size) - 0.5
    b2 = np.random.rand(1, output_size) - 0.5
    return W1, b1, W2, b2

# Propagación hacia adelante
def forward_propagation(X, W1, b1, W2, b2):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output = sigmoid(output_layer_input)
    return hidden_layer_output, output

# Backpropagation para actualizar pesos
def backpropagation(X, y, hidden_output, output, W1, W2, b1, b2, learning_rate):
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    return W1, b1, W2, b2

# Entrenamiento del modelo
def train(X, y, hidden_size=5, learning_rate=0.1, epochs=5000):
    input_size = X.shape[1]
    output_size = 1
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        hidden_output, output = forward_propagation(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = backpropagation(X, y, hidden_output, output, W1, W2, b1, b2, learning_rate)
        
        if epoch % 500 == 0:
            loss = np.mean(np.square(y - output))
            print(f"Epoch {epoch}: Loss = {loss:.5f}")
    
    return W1, b1, W2, b2

# Evaluación del modelo
def evaluate(X, y, W1, b1, W2, b2):
    _, output = forward_propagation(X, W1, b1, W2, b2)
    predictions = (output > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Predicción para una nueva máquina
def predict_machine(W1, b1, W2, b2):
    temperatura = float(input("Ingrese la temperatura: "))
    presion = float(input("Ingrese la presión: "))
    corriente = float(input("Ingrese la corriente: "))
    
    X_new = np.array([[temperatura, presion, corriente]])
    _, output = forward_propagation(X_new, W1, b1, W2, b2)
    
    print(f"La probabilidad de fallar es: {output[0][0]:.2f}%")
    prediction = (output > 0.5).astype(int)
    
    if prediction == 1:
        print("La máquina fallará.")
    else:
        print("La máquina no fallará.")

# Ejecutar el modelo
X, y = load_dataset()
train_size = int(0.6 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

W1, b1, W2, b2 = train(X_train, y_train)
evaluate(X_test, y_test, W1, b1, W2, b2)

# Predecir el estado de una nueva máquina
predict_machine(W1, b1, W2, b2)
