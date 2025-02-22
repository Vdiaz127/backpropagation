import numpy as np
import json

# 1. DATOS Y PREPROCESAMIENTO
# ---------------------------
# Leer datos desde el archivo JSON
with open('maquinas.json', 'r') as file:
    datos = json.load(file)

# Convertir a arrays numpy
X = np.array([[d['temperatura'], d['presion'], d['corriente']] for d in datos])
y = np.array([d['fallo'] for d in datos]).reshape(-1, 1)

# Normalización
indices = np.random.permutation(len(X))
train_idx = indices[:int(0.6*len(X))]
test_idx = indices[int(0.6*len(X)):]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

media = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - media) / std
X_test = (X_test - media) / std

# 2. ARQUITECTURA MLP OPTIMIZADA
# ------------------------------
class MLPIndustrial:
    def __init__(self):
        # Inicializa los pesos y sesgos de la red neuronal
        self.W1 = np.random.randn(3, 8) * np.sqrt(2/(3+8))
        self.b1 = np.zeros((1, 8))
        self.W2 = np.random.randn(8, 1) * np.sqrt(2/(8+1))
        self.b2 = np.zeros((1, 1))
    
    def _relu(self, x):
        # Función de activación ReLU
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        # Función de activación Sigmoide
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        return self._sigmoid(self.z2)
    
    def _loss(self, y_true, y_pred):
        # Calcula la pérdida (entropía cruzada)
        eps = 1e-8
        return -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))
    
    def entrenar(self, X, y, epochs=2000, lr=0.005):
        # Entrena la red neuronal usando retropropagación
        for epoch in range(epochs):
            # Propagación
            y_pred = self.forward(X)
            loss = self._loss(y, y_pred)
            
            # Retropropagación
            m = X.shape[0]
            dz2 = y_pred - y
            dw2 = (self.a1.T.dot(dz2)) / m
            db2 = np.sum(dz2, axis=0) / m
            
            dz1 = dz2.dot(self.W2.T) * (self.z1 > 0)
            dw1 = (X.T.dot(dz1)) / m
            db1 = np.sum(dz1, axis=0) / m
            
            # Actualización de pesos y sesgos
            self.W2 -= lr * dw2
            self.b2 -= lr * db2
            self.W1 -= lr * dw1
            self.b1 -= lr * db1
            
            if epoch % 400 == 0:
                print(f"Época {epoch}: Pérdida = {loss:.4f}")
    
    def predecir(self, X, umbral=0.5):
        # Realiza una predicción
        return (self.forward(X) > umbral).astype(int)

# 3. ENTRENAMIENTO
# ----------------
modelo = MLPIndustrial()
modelo.entrenar(X_train, y_train)

# 4. PREDICCIÓN
# -------------
def predecir_fallo(temperatura, presion, corriente):
    # Normaliza los datos de entrada
    datos = np.array([[temperatura, presion, corriente]])
    datos_norm = (datos - media) / std
    
    # Realiza la predicción
    prob = modelo.forward(datos_norm)[0][0]
    print(prob)
    return "Falla" if prob > 0.5 else "No Falla"

# Ejemplo de uso con tus valores críticos
print("\nPrueba crítica:")
print(predecir_fallo(100, 30, 80)) 