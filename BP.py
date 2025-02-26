import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Establecer una semilla para garantizar que los resultados sean reproducibles
np.random.seed(0)

# Definición de funciones de activación utilizadas en la red neuronal
def sigmoid(x):
    """Función sigmoide: transforma valores en un rango de 0 a 1, útil para probabilidades"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Función ReLU: devuelve 0 para valores negativos y el valor original si es positivo"""
    return np.maximum(0, x)

def tanh(x):
    """Función tangente hiperbólica: transforma valores en un rango de -1 a 1"""
    return np.tanh(x)

class RedDeteccionFallas:
    def __init__(self, datos_entrada, respuestas_reales, neuronas_ocultas=4, activacion='sigmoid'):
        """
        Inicializa la red neuronal con los datos de entrenamiento y parámetros configurables.
        - datos_entrada: Matriz de datos de entrada (características normalizadas)
        - respuestas_reales: Vector de etiquetas reales (fallo=1, no fallo=0)
        - neuronas_ocultas: Cantidad de neuronas en la capa oculta
        - activacion: Tipo de función de activación para ambas capas (oculta y salida)
        """
        print("FUNCIÓN DE ACTIVACIÓN: " + activacion)

        # Almacenar datos de entrenamiento y etiquetas
        self.datos_entrenamiento = datos_entrada  # Datos de entrada para entrenamiento
        self.respuestas_reales = respuestas_reales # Etiquetas reales asociadas
        self.neuronas_ocultas = neuronas_ocultas # Número de neuronas en la capa oculta
        self.caracteristicas_entrada = datos_entrada.shape[1] # Número de características (3: temperatura, presión, corriente)
        self.activacion = activacion # Función de activación
        
        # Inicialización de pesos y sesgos con valores aleatorios pequeños para evitar saturación inicial
        self.pesos_entrada_oculta = np.random.randn(neuronas_ocultas, self.caracteristicas_entrada) * 0.1  # Pesos de entrada a oculta
        self.sesgos_oculta = np.zeros(neuronas_ocultas) # Sesgos de la capa oculta (inician en 0)
        self.pesos_oculta_salida = np.random.randn(neuronas_ocultas) * 0.1  # Pesos de oculta a salida
        self.sesgo_salida = 0.0 # Sesgo de la capa de salida (inicia en 0)
        
        # Listas para almacenar métricas de entrenamiento
        self.historial_error = []  # Registro del error acumulado por época
        self.historial_accuracy = []  # Registro de la precisión por época

    def _activacion(self, x, tipo):
        """Aplica la función de activación especificada al valor de entrada x"""
        if tipo == 'sigmoid':
            return sigmoid(x)  # Aplica sigmoide para valores entre 0 y 1
        elif tipo == 'relu':
            return relu(x)  # Aplica ReLU para valores no negativos
        elif tipo == 'tanh':
            return tanh(x)  # Aplica tanh para valores entre -1 y 1
        else:
            raise ValueError("Función de activación no soportada")

    def _derivada_activacion(self, x, tipo):
        """Calcula la derivada de la función de activación para usar en retropropagación"""
        if tipo == 'sigmoid':
            return x * (1 - x)  # Derivada de sigmoide basada en la salida
        elif tipo == 'relu':
            return (x > 0).astype(float)  # Derivada de ReLU: 1 si x > 0, 0 si no
        elif tipo == 'tanh':
            return 1 - x**2  # Derivada de tanh basada en la salida
        else:
            raise ValueError("Derivada no implementada para esta activación")

    def _interpretar_salida(self, valor_salida, tipo):
        """
        Interpreta la salida cruda de la red para convertirla en una predicción binaria (0 o 1).
        - valor_salida: Valor de salida de la capa final
        - tipo: Tipo de función de activación usada
        """
        if tipo == 'sigmoid':
            return 1 if valor_salida >= 0.5 else 0  # Umbral de 0.5 para clasificar como fallo/no fallo
        elif tipo == 'relu':
            return 1 if valor_salida > 0.5 else 0   # Umbral arbitrario para ReLU
        elif tipo == 'tanh':
            return 1 if valor_salida > 0 else 0     # Umbral en 0 para el rango -1 a 1 de tanh
        else:
            raise ValueError("Interpretación no definida para esta activación")

    def entrenar(self, ritmo_aprendizaje=0.01, vueltas_entrenamiento=1000, verbose=True):
        """
        Entrena la red neuronal utilizando el algoritmo de retropropagación.
        - ritmo_aprendizaje: Tasa que controla el tamaño de las actualizaciones de pesos
        - vueltas_entrenamiento: Número de iteraciones completas sobre el conjunto de entrenamiento
        - verbose: Si es True, muestra métricas cada 10 épocas
        """
        for epoca in range(vueltas_entrenamiento):
            error_acumulado = 0.0  # Error total acumulado en esta época
            predicciones_correctas = 0  # Contador de predicciones correctas
            
            # Iterar sobre cada muestra del conjunto de entrenamiento
            for i in range(self.datos_entrenamiento.shape[0]):
                # Propagación hacia adelante
                entrada = self.datos_entrenamiento[i] # Extraer la muestra de entrada actual

                # Calcular la salida de la capa oculta
                combinacion_lineal_oculta = np.dot(self.pesos_entrada_oculta, entrada) + self.sesgos_oculta # Combinación lineal
                salida_oculta = self._activacion(combinacion_lineal_oculta, self.activacion)  # Aplicar función de activación

                # Calcular la salida de la capa de salida
                combinacion_lineal_salida = np.dot(salida_oculta, self.pesos_oculta_salida) + self.sesgo_salida # Combinación lineal
                valor_salida = self._activacion(combinacion_lineal_salida, self.activacion) # Aplicar función de activación
                
                # --- Calcular error y precisión ---
                error = (valor_salida - self.respuestas_reales[i])**2  # Error cuadrático entre predicción y etiqueta real
                error_acumulado += error  # Acumular error para la época
                etiqueta_predicha = self._interpretar_salida(valor_salida, self.activacion)  # Convertir a predicción binaria
                if etiqueta_predicha == self.respuestas_reales[i]:
                    predicciones_correctas += 1  # Incrementar contador si la predicción es correcta
                
                # Retropropagación
                # Calcular el gradiente del error en la capa de salida
                gradiente_error_salida = (valor_salida - self.respuestas_reales[i]) * self._derivada_activacion(valor_salida, self.activacion)

                # Calcular el gradiente del error en la capa oculta
                gradiente_error_oculta = gradiente_error_salida * self.pesos_oculta_salida * self._derivada_activacion(salida_oculta, self.activacion)
                
                # Actualizar pesos y sesgos de la capa oculta para cada neurona
                self.pesos_oculta_salida -= ritmo_aprendizaje * gradiente_error_salida * salida_oculta
                self.sesgo_salida -= ritmo_aprendizaje * gradiente_error_salida
                
                for neurona in range(self.neuronas_ocultas):
                    self.pesos_entrada_oculta[neurona] -= ritmo_aprendizaje * gradiente_error_oculta[neurona] * entrada
                    self.sesgos_oculta[neurona] -= ritmo_aprendizaje * gradiente_error_oculta[neurona]
            
            # --- Registrar métricas de la época ---
            self.historial_error.append(error_acumulado) # Guardar error acumulado
            accuracy = predicciones_correctas / self.datos_entrenamiento.shape[0]  # Calcular precisión como proporción de aciertos
            self.historial_accuracy.append(accuracy) # Guardar precisión

            # Mostrar progreso si verbose está activado
            if verbose and epoca % 10 == 0:
                print(f"Época {epoca}: Error = {error_acumulado:.4f}, Accuracy = {accuracy:.2%}")

    def predecir(self, entrada):
        """
        Realiza una predicción para una muestra de entrada específica.
        - entrada: Vector de características de una máquina
        """

        # Propagación hacia adelante para la predicción
        combinacion_lineal_oculta = np.dot(self.pesos_entrada_oculta, entrada) + self.sesgos_oculta
        salida_oculta = self._activacion(combinacion_lineal_oculta, self.activacion)
        combinacion_lineal_salida = np.dot(salida_oculta, self.pesos_oculta_salida) + self.sesgo_salida
        valor_salida = self._activacion(combinacion_lineal_salida, self.activacion)
        return self._interpretar_salida(valor_salida, self.activacion) # Devolver predicción binaria

    def evaluar(self, caracteristicas, etiquetas):
        """
        Evalúa el rendimiento de la red en un conjunto de prueba.
        - caracteristicas: Matriz de datos de prueba
        - etiquetas: Vector de etiquetas reales de prueba
        Retorna un diccionario con métricas de evaluación.
        """
        predicciones = np.array([self.predecir(x) for x in caracteristicas])  # Generar predicciones para todas las muestras
        etiquetas = np.array(etiquetas)  # Asegurar que las etiquetas sean un array NumPy
        
        # Calcular precisión global
        accuracy = np.mean(predicciones == etiquetas)
        
        # Calcular elementos de la matriz de confusión
        verdaderos_negativos = np.sum((etiquetas == 0) & (predicciones == 0))  # No fallo bien predicho
        falsos_positivos = np.sum((etiquetas == 0) & (predicciones == 1))  # No fallo predicho como fallo
        falsos_negativos = np.sum((etiquetas == 1) & (predicciones == 0))  # Fallo predicho como no fallo
        verdaderos_positivos = np.sum((etiquetas == 1) & (predicciones == 1))  # Fallo bien predicho
        
        # Calcular métricas adicionales
        precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
        recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,  # Precisión global
            'precision': precision,  # Proporción de positivos correctos
            'recall': recall,  # Proporción de fallos detectados
            'f1': f1,  # Media armónica de precisión y recall
            'matriz_confusion': (verdaderos_negativos, falsos_positivos, falsos_negativos, verdaderos_positivos)  # Matriz de confusión
        }

# --- Procesamiento de datos ---
# Cargar los datos desde el archivo JSON
with open("maquinas.json", "r") as f:
    datos = json.load(f)

# Extraer características y etiquetas del archivo JSON
X = np.array([[m['temperatura'], m['presion'], m['corriente']] for m in datos])  # Matriz de características
y = np.array([m['fallo'] for m in datos])  # Vector de etiquetas (0 o 1)


# Normalizar datos al rango [0, 1] usando min-max
valores_minimos = X.min(axis=0)  # Valores mínimos por característica
valores_maximos = X.max(axis=0)  # Valores máximos por característica
X_normalizado = (X - valores_minimos) / (valores_maximos - valores_minimos)  # Normalización

# Dividir datos en entrenamiento y validación
datos_entrenamiento, datos_validacion, respuestas_entrenamiento, respuestas_validacion = train_test_split(
    X_normalizado, y, test_size=0.4, random_state=0)

# --- Entrenamiento y evaluación ---
activaciones = ['sigmoid', 'relu', 'tanh']  # Lista de funciones de activación a probar
for activacion in activaciones:
    print(f"\n=== Evaluando con activación: {activacion} ===")
    
    # Crear y entrenar la red con la activación seleccionada
    red = RedDeteccionFallas(datos_entrenamiento, respuestas_entrenamiento, neuronas_ocultas=6, activacion=activacion)
    red.entrenar(ritmo_aprendizaje=0.01, vueltas_entrenamiento=300) # Usar tasa baja como ejemplo
    
    # Evaluar en conjuntos de entrenamiento y validación
    resultados_entrenamiento = red.evaluar(datos_entrenamiento, respuestas_entrenamiento)
    resultados_validacion = red.evaluar(datos_validacion, respuestas_validacion)
    
    # Imprimir resultados de entrenamiento
    print("\nResultados Entrenamiento:")
    print(f"Accuracy: {resultados_entrenamiento['accuracy']:.2%}")
    print(f"Precision: {resultados_entrenamiento['precision']:.2%}")
    print(f"Recall: {resultados_entrenamiento['recall']:.2%}")
    print(f"f1: {resultados_entrenamiento['f1']:.4f}")
    print(f"Matriz de Confusión (TN, FP, FN, TP): {resultados_entrenamiento['matriz_confusion']}")
    
    # Imprimir resultados de validación
    print("\nResultados Validación:")
    print(f"Accuracy: {resultados_validacion['accuracy']:.2%}")
    print(f"Precision: {resultados_validacion['precision']:.2%}")
    print(f"Recall: {resultados_validacion['recall']:.2%}")
    print(f"Matriz de Confusión (TN, FP, FN, TP): {resultados_validacion['matriz_confusion']}")
    
    # Visualización de métricas
    plt.figure(figsize=(12, 5))
    
    # Gráfico del error
    plt.subplot(1, 2, 1)
    plt.plot(red.historial_error)
    plt.title(f'Curva de Aprendizaje ({activacion})')
    plt.xlabel('Época')
    plt.ylabel('Error')
    
    # Gráfico de la precisión
    plt.subplot(1, 2, 2)
    plt.plot(red.historial_accuracy)
    plt.title(f'Precisión durante Entrenamiento ({activacion})')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    
    # Ejemplo de predicción con una nueva muestra
    nueva_maquina = np.array([85, 12, 23]) # Nueva máquina con valores de sensores
    nueva_maquina_normalizada = (nueva_maquina - valores_minimos) / (valores_maximos - valores_minimos) #normalizar
    prediccion = red.predecir(nueva_maquina_normalizada) 
    print(f"Predicción para {nueva_maquina}: {'Falla' if prediccion == 1 else 'Normal'}")