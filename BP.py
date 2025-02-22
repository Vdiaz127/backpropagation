import numpy as np
import json

# Para reproducibilidad
np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# Carga y Preprocesamiento de Datos
# -------------------------
# Se asume que el archivo 'maquinas.json' contiene una lista de objetos con las claves:
# "temperatura", "presion", "corriente" y "fallo"
with open("maquinas.json", "r") as f:
    data = json.load(f)

# Extraer características y etiquetas
X = []
Y = []
for entry in data:
    # Se utilizan tres características: temperatura, presion y corriente
    X.append([entry["temperatura"], entry["presion"], entry["corriente"]])
    Y.append(entry["fallo"])

X = np.array(X)
Y = np.array(Y)

# Normalización: se aplica min-max scaling a las características
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# -------------------------
# Definición de la Red Neuronal
# -------------------------
# La red tendrá:
# - Capa de entrada: 3 neuronas (una por cada característica)
# - Capa oculta: 4 neuronas
# - Capa de salida: 1 neurona (clasificación binaria: fallo/no fallo)
class RedNeuronal:
    def __init__(self, x, y, n_hidden=4):
        # Datos de entrenamiento
        self.x = x  # Características de entrada
        self.y = y  # Etiquetas de salida
        self.n_hidden = n_hidden  # Número de neuronas en la capa oculta
        self.n_features = x.shape[1]  # Número de características de entrada
        # Inicialización aleatoria de pesos y sesgos (centrados en 0)
        # Pesos de la capa oculta: matriz de dimensiones (n_hidden, n_features)
        self.pesos1 = np.random.rand(n_hidden, self.n_features) - 0.5
        print("Pesos 1")
        print(self.pesos1)
        # Sesgos para la capa oculta: vector de tamaño n_hidden
        self.sesgos1 = np.random.rand(n_hidden) - 0.5
        print("Sesgos 1")
        print(self.sesgos1)
        # Pesos de la capa de salida: vector de tamaño n_hidden (cada neurona oculta aporta a la salida)
        self.pesos2 = np.random.rand(n_hidden) - 0.5
        print("Pesos 2")
        print(self.pesos2)
        # Sesgo para la neurona de salida: escalar
        self.sesgos2 = np.random.rand(1) - 0.5
        print("Sesgo 2")
        print(self.sesgos2)

    def entrenamiento(self, tasa_aprendizaje=0.1, epocas=1000):
        for epoch in range(epocas):
            error_total = 0.0
            # Iterar sobre cada muestra del dataset
            for i in range(self.x.shape[0]):
                # -------------------------
                # Propagación hacia adelante
                # -------------------------
                # Capa oculta: calcular la salida de cada neurona
                hidden_outputs = np.zeros(self.n_hidden)
                for j in range(self.n_hidden):
                    # Suma ponderada de las entradas + sesgo para la neurona j
                    z = np.dot(self.x[i], self.pesos1[j]) + self.sesgos1[j]
                    hidden_outputs[j] = sigmoid(z)
                # Capa de salida: combinación lineal de las salidas ocultas + sesgo
                suma_s = np.dot(hidden_outputs, self.pesos2) + self.sesgos2[0]
                y_pred = sigmoid(suma_s)
                
                # Cálculo del error cuadrático para la muestra i
                error = 0.5 * (self.y[i] - y_pred) ** 2
                error_total += error
                
                # -------------------------
                # Backpropagation
                # -------------------------
                # Capa de salida: calcular el delta (error) de la neurona de salida
                delta_output = (y_pred - self.y[i]) * y_pred * (1 - y_pred)
                
                # Actualización de pesos y sesgo de la capa de salida
                for j in range(self.n_hidden):
                    grad_pesos2 = delta_output * hidden_outputs[j]
                    self.pesos2[j] -= tasa_aprendizaje * grad_pesos2
                print("gradiente de peso2")
                print(grad_pesos2)
                self.sesgos2[0] -= tasa_aprendizaje * delta_output
                
                # Capa oculta: propagar el error hacia atrás
                for j in range(self.n_hidden):
                    delta_hidden = delta_output * self.pesos2[j] * hidden_outputs[j] * (1 - hidden_outputs[j])
                    # Actualizar pesos de la neurona oculta j
                    for k in range(self.n_features):
                        self.pesos1[j, k] -= tasa_aprendizaje * delta_hidden * self.x[i][k]
                    # Actualizar el sesgo de la neurona oculta j
                    self.sesgos1[j] -= tasa_aprendizaje * delta_hidden
            
            # Imprimir el error total cada 100 épocas
            if epoch % 100 == 0:
                print(f'Época {epoch}: Error total = {error_total}')

    def clasificacion(self, entrada):
        # Propagación hacia adelante para evaluar una nueva instancia
        hidden_outputs = np.zeros(self.n_hidden)
        for j in range(self.n_hidden):
            z = np.dot(entrada, self.pesos1[j]) + self.sesgos1[j]
            hidden_outputs[j] = sigmoid(z)
        suma_s = np.dot(hidden_outputs, self.pesos2) + self.sesgos2[0]
        y_pred = sigmoid(suma_s)
        print("resultado de predicion:" + str(y_pred))
        return round(y_pred)

# -------------------------
# Entrenamiento y Evaluación
# -------------------------
# Crear la red neuronal usando los datos normalizados
red_neuronal = RedNeuronal(X_norm, Y, n_hidden=4)
red_neuronal.entrenamiento(tasa_aprendizaje=0.1, epocas=1000)

# Ejemplo de clasificación:
# Definir una nueva instancia con valores originales de temperatura, presión y corriente.
nueva_instancia = [90, 10, 25]  # Ejemplo: valores de temperatura, presión y corriente
# Normalizar la nueva instancia usando los mismos parámetros del dataset original
nueva_instancia_norm = (np.array(nueva_instancia) - X_min) / (X_max - X_min)
resultado = red_neuronal.clasificacion(nueva_instancia_norm)
print(f'\nResultado de la clasificación para {nueva_instancia}: {resultado}')
