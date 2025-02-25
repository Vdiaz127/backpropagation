import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configurar semilla para reproducibilidad
np.random.seed(0)

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

class RedDeteccionFallas:
    def __init__(self, datos_entrada, respuestas_reales, neuronas_ocultas=4,
                 activacion_oculta='sigmoid', activacion_salida='sigmoid'):
        # Configuración de la red
        print("FUNCIÓN DE ACTIVACIÓN OCULTA: " + activacion_oculta)
        print("FUNCIÓN DE ACTIVACIÓN SALIDA: " + activacion_salida)
        self.datos_entrenamiento = datos_entrada
        self.respuestas_reales = respuestas_reales
        self.neuronas_ocultas = neuronas_ocultas
        self.caracteristicas_entrada = datos_entrada.shape[1]
        self.activacion_oculta = activacion_oculta
        self.activacion_salida = activacion_salida
        
        # Inicialización de parámetros
        self.pesos_entrada_oculta = np.random.randn(neuronas_ocultas, self.caracteristicas_entrada) * 0.1
        self.sesgos_oculta = np.zeros(neuronas_ocultas)
        self.pesos_oculta_salida = np.random.randn(neuronas_ocultas) * 0.1
        self.sesgo_salida = 0.0
        
        # Historial de entrenamiento
        self.historial_error = []
        self.historial_accuracy = []

    def _activacion(self, x, tipo):
        if tipo == 'sigmoid':
            return sigmoid(x)
        elif tipo == 'relu':
            return relu(x)
        elif tipo == 'tanh':
            return tanh(x)
        else:
            raise ValueError("Función de activación no soportada")

    def _derivada_activacion(self, x, tipo):
        if tipo == 'sigmoid':
            return x * (1 - x)
        elif tipo == 'relu':
            return (x > 0).astype(float)
        elif tipo == 'tanh':
            return 1 - x**2
        else:
            raise ValueError("Derivada no implementada")

    def _interpretar_salida(self, a_salida, tipo):
        """Interpreta la salida según la función de activación para clasificación binaria"""
        if tipo == 'sigmoid':
            return 1 if a_salida >= 0.5 else 0  # Umbral en 0.5 para sigmoid
        elif tipo == 'relu':
            return 1 if a_salida > 0.5 else 0   # Umbral arbitrario para ReLU
        elif tipo == 'tanh':
            return 1 if a_salida > 0 else 0     # Umbral en 0 para tanh (-1 a 1)
        else:
            raise ValueError("Interpretación no definida para esta activación")

    def entrenar(self, ritmo_aprendizaje=0.01, vueltas_entrenamiento=1000, verbose=True):
        for vuelta in range(vueltas_entrenamiento):
            error_acumulado = 0.0
            correctos = 0
            
            for i in range(self.datos_entrenamiento.shape[0]):
                # Propagación hacia adelante
                entrada = self.datos_entrenamiento[i]
                
                # Capa oculta
                z_oculta = np.dot(self.pesos_entrada_oculta, entrada) + self.sesgos_oculta
                a_oculta = self._activacion(z_oculta, self.activacion_oculta)
                
                # Capa salida
                z_salida = np.dot(a_oculta, self.pesos_oculta_salida) + self.sesgo_salida
                a_salida = self._activacion(z_salida, self.activacion_salida)
                
                # Cálculo de error
                error = (a_salida - self.respuestas_reales[i])**2
                error_acumulado += error
                prediccion = self._interpretar_salida(a_salida, self.activacion_salida)
                if prediccion == self.respuestas_reales[i]:
                    correctos += 1
                
                # Retropropagación
                delta_salida = (a_salida - self.respuestas_reales[i]) * self._derivada_activacion(a_salida, self.activacion_salida)
                delta_oculta = delta_salida * self.pesos_oculta_salida * self._derivada_activacion(a_oculta, self.activacion_oculta)
                
                # Actualizar pesos
                self.pesos_oculta_salida -= ritmo_aprendizaje * delta_salida * a_oculta
                self.sesgo_salida -= ritmo_aprendizaje * delta_salida
                
                for neurona in range(self.neuronas_ocultas):
                    self.pesos_entrada_oculta[neurona] -= ritmo_aprendizaje * delta_oculta[neurona] * entrada
                    self.sesgos_oculta[neurona] -= ritmo_aprendizaje * delta_oculta[neurona]
            
            # Guardar métricas
            self.historial_error.append(error_acumulado)
            accuracy = correctos / self.datos_entrenamiento.shape[0]
            self.historial_accuracy.append(accuracy)
            
            if verbose and vuelta % 10 == 0:
                print(f"Época {vuelta}: Error = {error_acumulado:.4f}, Accuracy = {accuracy:.2%}")

    def predecir(self, entrada):
        z_oculta = np.dot(self.pesos_entrada_oculta, entrada) + self.sesgos_oculta
        a_oculta = self._activacion(z_oculta, self.activacion_oculta)
        z_salida = np.dot(a_oculta, self.pesos_oculta_salida) + self.sesgo_salida
        a_salida = self._activacion(z_salida, self.activacion_salida)
        return self._interpretar_salida(a_salida, self.activacion_salida)

    def evaluar(self, caracteristicas, etiquetas):
        predicciones = np.array([self.predecir(x) for x in caracteristicas])
        etiquetas = np.array(etiquetas)
        
        accuracy = np.mean(predicciones == etiquetas)
        tn = np.sum((etiquetas == 0) & (predicciones == 0))
        fp = np.sum((etiquetas == 0) & (predicciones == 1))
        fn = np.sum((etiquetas == 1) & (predicciones == 0))
        tp = np.sum((etiquetas == 1) & (predicciones == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matriz_confusion': (tn, fp, fn, tp)
        }

# Procesamiento de datos
with open("maquinas.json", "r") as f:
    datos = json.load(f)

X = np.array([[m['temperatura'], m['presion'], m['corriente']] for m in datos])
y = np.array([m['fallo'] for m in datos])

val_min = X.min(axis=0)
val_max = X.max(axis=0)
X_norm = (X - val_min) / (val_max - val_min)

X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.4, random_state=0)

# Entrenamiento y evaluación con diferentes combinaciones
activaciones = ['sigmoid', 'relu', 'tanh']
for activacion in activaciones:
    print(f"\n=== Evaluando con activación de salida: {activacion} ===")
    red = RedDeteccionFallas(X_train, y_train, neuronas_ocultas=6, 
                            activacion_oculta=activacion, activacion_salida=activacion)
    red.entrenar(ritmo_aprendizaje=0.0000011, vueltas_entrenamiento=300)

    resultados_train = red.evaluar(X_train, y_train)
    resultados_val = red.evaluar(X_val, y_val)

    print("\nResultados Entrenamiento:")
    print(f"Accuracy: {resultados_train['accuracy']:.2%}")
    print(f"Precision: {resultados_train['precision']:.2%}")
    print(f"Recall: {resultados_train['recall']:.2%}")
    print(f"f1: {resultados_train['f1']:.4f}")
    print(f"Matriz de Confusión (TN, FP, FN, TP): {resultados_train['matriz_confusion']}")

    print("\nResultados Validación:")
    print(f"Accuracy: {resultados_val['accuracy']:.2%}")
    print(f"Precision: {resultados_val['precision']:.2%}")
    print(f"Recall: {resultados_val['recall']:.2%}")
    print(f"Matriz de Confusión (TN, FP, FN, TP): {resultados_val['matriz_confusion']}")

    # Visualización
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(red.historial_error)
    plt.title(f'Curva de Aprendizaje ({activacion} en salida)')
    plt.xlabel('Época')
    plt.ylabel('Error')

    plt.subplot(1, 2, 2)
    plt.plot(red.historial_accuracy)
    plt.title(f'Precisión durante Entrenamiento ({activacion} en salida)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

    # Ejemplo de predicción
    nueva_maquina = np.array([85, 12, 23])
    nueva_norm = (nueva_maquina - val_min) / (val_max - val_min)
    prediccion = red.predecir(nueva_norm)
    print(f"Predicción para {nueva_maquina}: {'Falla 🔴' if prediccion == 1 else 'Normal ✅'}")