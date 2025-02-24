import numpy as np
import json

# Configurar semilla para obtener mismos resultados siempre
np.random.seed(0)

# Función de activación sigmoide (transforma números a rango 0-1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# 1. CARGAR Y PREPARAR DATOS
# -------------------------
# Leer archivo con datos de máquinas
with open("maquinas.json", "r") as archivo:
    datos_maquinas = json.load(archivo)

# Listas para almacenar características y respuestas correctas
caracteristicas = []  # [temperatura, presión, corriente]
etiquetas_reales = []  # [1 (falla) o 0 (no falla)]

for maquina in datos_maquinas:
    # Agregar medidas de la máquina
    caracteristicas.append([
        maquina["temperatura"],
        maquina["presion"],
        maquina["corriente"]
    ])
    # Agregar si tuvo falla (1) o no (0)
    etiquetas_reales.append(maquina["fallo"])

# Convertir a arrays de NumPy para operaciones matemáticas
caracteristicas = np.array(caracteristicas)
etiquetas_reales = np.array(etiquetas_reales)

# Normalizar datos (convertir valores a rango 0-1)
val_min = caracteristicas.min(axis=0)
val_max = caracteristicas.max(axis=0)
caracteristicas_normalizadas = (caracteristicas - val_min) / (val_max - val_min)

# -------------------------
# 2. CREAR RED NEURONAL
# -------------------------
class RedDeteccionFallas:
    def __init__(self, datos_entrada, respuestas_reales, neuronas_ocultas=4):
        # Datos para entrenamiento
        self.datos_entrenamiento = datos_entrada    # Características normalizadas
        self.respuestas_reales = respuestas_reales    # Respuestas correctas (0 o 1)
        self.neuronas_ocultas = neuronas_ocultas      # Número de neuronas en capa oculta
        self.caracteristicas_entrada = datos_entrada.shape[1]  # Número de características (3)
        
        # Inicializar parámetros aleatoriamente (pesos y sesgos)
        # Pesos entrada -> oculta (3 características -> 4 neuronas)
        self.pesos_entrada_oculta = np.random.rand(neuronas_ocultas, self.caracteristicas_entrada) - 0.5
        print("Pesos iniciales entrada-oculta:")
        print(self.pesos_entrada_oculta)
        
        # Sesgos capa oculta (1 por neurona oculta)
        self.sesgos_oculta = np.random.rand(neuronas_ocultas) - 0.5
        print("\nSesgos iniciales capa oculta:")
        print(self.sesgos_oculta)
        
        # Pesos oculta -> salida (4 neuronas -> 1 salida)
        self.pesos_oculta_salida = np.random.rand(neuronas_ocultas) - 0.5
        print("\nPesos iniciales oculta-salida:")
        print(self.pesos_oculta_salida)
        
        # Sesgo capa salida (generar directamente un escalar)
        self.sesgo_salida = np.random.rand() - 0.5
        print("\nSesgo inicial capa salida:")
        print(self.sesgo_salida)

    def entrenar(self, ritmo_aprendizaje=0.1, vueltas_entrenamiento=1000):
        for vuelta in range(vueltas_entrenamiento):
            error_acumulado = 0.0
            
            # Procesar cada máquina del dataset
            for i in range(self.datos_entrenamiento.shape[0]):
                # ----------------------------------
                # PASO 1: Propagación hacia adelante
                # ----------------------------------
                # Calcular salida de la capa oculta
                salida_oculta = np.zeros(self.neuronas_ocultas)
                for neurona in range(self.neuronas_ocultas):
                    # Sumar: (entradas * pesos) + sesgo
                    suma_ponderada = (np.dot(self.datos_entrenamiento[i],
                                               self.pesos_entrada_oculta[neurona])
                                      + float(self.sesgos_oculta[neurona]))
                    # Aplicar función de activación
                    salida_oculta[neurona] = sigmoid(suma_ponderada)
                
                # Calcular predicción final
                suma_salida = np.dot(salida_oculta, self.pesos_oculta_salida) + self.sesgo_salida
                prediccion = float(sigmoid(suma_salida))
                
                # Calcular error (diferencia entre predicción y realidad)
                error = 0.5 * (self.respuestas_reales[i] - prediccion) ** 2
                error_acumulado += error
                
                # ----------------------------------
                # PASO 2: Retropropagación de errores
                # ----------------------------------
                # Calcular error en la capa de salida
                error_salida = (prediccion - self.respuestas_reales[i]) * prediccion * (1 - prediccion)
                
                # Ajustar pesos y sesgos de SALIDA
                for neurona in range(self.neuronas_ocultas):
                    ajuste_peso = error_salida * salida_oculta[neurona]
                    self.pesos_oculta_salida[neurona] = (float(self.pesos_oculta_salida[neurona])
                                                         - ritmo_aprendizaje * ajuste_peso)
                
                # Ajustar sesgo de salida
                self.sesgo_salida = self.sesgo_salida - ritmo_aprendizaje * error_salida
                
                # Ajustar pesos y sesgos de la capa OCULTA
                for neurona in range(self.neuronas_ocultas):
                    error_oculto = (error_salida *
                                    float(self.pesos_oculta_salida[neurona]) *
                                    salida_oculta[neurona] *
                                    (1 - salida_oculta[neurona]))
                    
                    # Ajustar pesos ENTRADA -> OCULTA
                    for caracteristica in range(self.caracteristicas_entrada):
                        ajuste = ritmo_aprendizaje * error_oculto * self.datos_entrenamiento[i][caracteristica]
                        self.pesos_entrada_oculta[neurona, caracteristica] = (
                            float(self.pesos_entrada_oculta[neurona, caracteristica]) - ajuste)
                    
                    # Ajustar sesgo de la capa oculta
                    self.sesgos_oculta[neurona] = (float(self.sesgos_oculta[neurona])
                                                   - ritmo_aprendizaje * error_oculto)
            
            # Mostrar progreso cada 100 vueltas
            if vuelta % 100 == 0:
                print(f'Vuelta {vuelta}: Error acumulado = {error_acumulado}')

    def predecir(self, nueva_medida):
        # Calcular salida de la capa oculta para la nueva medida
        salida_oculta = np.zeros(self.neuronas_ocultas)
        for neurona in range(self.neuronas_ocultas):
            suma_ponderada = (np.dot(nueva_medida, self.pesos_entrada_oculta[neurona])
                              + float(self.sesgos_oculta[neurona]))
            salida_oculta[neurona] = sigmoid(suma_ponderada)
        
        # Calcular la predicción final
        suma_salida = np.dot(salida_oculta, self.pesos_oculta_salida) + self.sesgo_salida
        probabilidad_falla = float(sigmoid(suma_salida))
        print(f"Probabilidad de falla calculada: {probabilidad_falla}")
        
        # Redondear a 0 (no falla) o 1 (falla)
        return round(probabilidad_falla)

# -------------------------
# 3. ENTRENAR Y USAR LA RED
# -------------------------
# Crear red neuronal con 4 neuronas ocultas
detector_fallas = RedDeteccionFallas(caracteristicas_normalizadas, etiquetas_reales, 4)

# Entrenar durante 1000 vueltas con ritmo de aprendizaje 0.1
detector_fallas.entrenar(ritmo_aprendizaje=0.1, vueltas_entrenamiento=1000)

# Ejemplo de predicción para una nueva máquina
nueva_maquina = [90, 10, 25]  # Valores originales (no normalizados)

# Normalizar usando los mismos parámetros que los datos originales
nueva_maquina_normalizada = (np.array(nueva_maquina) - val_min) / (val_max - val_min)

# Predecir si tendrá falla
resultado = detector_fallas.predecir(nueva_maquina_normalizada)
print(f'\nPara la máquina con valores {nueva_maquina}:')
print('¡ALERTA DE FALLA! 🔴' if resultado == 1 else 'Estado normal ✅')
