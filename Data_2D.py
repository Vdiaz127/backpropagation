import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración inicial
np.random.seed(42)

# Generar datos sintéticos
# Claro, te explico cómo se está segmentando la información en el código proporcionado.

# El código genera datos sintéticos para tres variables: temperatura, presion y corriente. Para cada una de estas variables, se utilizan distribuciones normales (np.random.normal) para simular diferentes segmentos de datos. Aquí está el desglose:

# Segmentación de la Información
# Temperatura:

# np.random.normal(50, 10, 50): Genera 50 valores con una media de 50 y una desviación estándar de 10. Estos representan temperaturas "normales".
# np.random.normal(85, 5, 25): Genera 25 valores con una media de 85 y una desviación estándar de 5. Estos representan temperaturas "altas".
# np.random.normal(90, 8, 25): Genera 25 valores con una media de 90 y una desviación estándar de 8. Estos representan temperaturas "muy altas".
# Presión:

# np.random.normal(100, 5, 50): Genera 50 valores con una media de 100 y una desviación estándar de 5. Estos representan presiones "óptimas".
# np.random.normal(70, 8, 25): Genera 25 valores con una media de 70 y una desviación estándar de 8. Estos representan presiones "bajas".
# np.random.normal(60, 10, 25): Genera 25 valores con una media de 60 y una desviación estándar de 10. Estos representan presiones "muy bajas".
# Corriente:

# np.random.normal(15, 2, 50): Genera 50 valores con una media de 15 y una desviación estándar de 2. Estos representan corrientes "normales".
# np.random.normal(25, 3, 25): Genera 25 valores con una media de 25 y una desviación estándar de 3. Estos representan corrientes "altas".
# np.random.normal(5, 1.5, 25): Genera 25 valores con una media de 5 y una desviación estándar de 1.5. Estos representan corrientes "bajas".
# Uso de np.concatenate
# La función np.concatenate se utiliza para unir los diferentes segmentos de datos generados en un solo array para cada variable. Por ejemplo, para temperatura, se concatenan los tres arrays generados (normales, altas y muy altas) en un solo array.

# Resumen de los Valores
# Cada tripleta de valores en np.random.normal(valor1, valor2, valor3) representa:

# valor1: La media de la distribución normal.
# valor2: La desviación estándar de la distribución normal.
# valor3: El número de valores a generar.
# Espero que esto aclare cómo se está segmentando la información y qué representa cada valor en las tripletas.

data = {
    'temperatura': np.concatenate([
        np.random.normal(50, 10, 50),  # Normales
        np.random.normal(85, 5, 25),    # Altas
        np.random.normal(90, 8, 25)     # Muy altas
    ]),
    'presion': np.concatenate([
        np.random.normal(100, 5, 50),   # Óptimas
        np.random.normal(70, 8, 25),    # Bajas
        np.random.normal(60, 10, 25)    # Muy bajas
    ]),
    'corriente': np.concatenate([
        np.random.normal(15, 2, 50),   # Normales
        np.random.normal(25, 3, 25),    # Altas
        np.random.normal(5, 1.5, 25)    # Bajas
    ])
}

df = pd.DataFrame(data)

# Crear etiquetas (1 = fallo próximo, 0 = normal)
df['fallo'] = 0  # Inicializar todas como normales

# Regla: 2 o más métricas en riesgo = fallo próximo
df.loc[
    ((df['temperatura'] > 80) & 
     (df['presion'] < 80)) |
    ((df['temperatura'] > 80) & 
     ((df['corriente'] > 20) | (df['corriente'] < 10))) |
    ((df['presion'] < 80) & 
     ((df['corriente'] > 20) | (df['corriente'] < 10))),
    'fallo'
] = 1

# >>> Guardar en JSON (nuevas líneas) <<<
df.to_json("maquinas.json", orient="records", indent=4)
print("¡JSON guardado correctamente como 'maquinas.json'!")

# Visualización
plt.figure(figsize=(15, 5))

# Temperatura vs Presión
plt.subplot(1, 3, 1)
plt.scatter(df['temperatura'][df['fallo'] == 0], df['presion'][df['fallo'] == 0], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(df['temperatura'][df['fallo'] == 1], df['presion'][df['fallo'] == 1], 
            c='red', label='Fallo próximo', alpha=0.6)
plt.title('Temperatura vs Presión')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Presión (psi)')
plt.axhline(80, color='gray', linestyle='--', alpha=0.5)  # Umbral presión
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)  # Umbral temperatura
plt.legend()

# Temperatura vs Corriente
plt.subplot(1, 3, 2)
plt.scatter(df['temperatura'][df['fallo'] == 0], df['corriente'][df['fallo'] == 0], 
            c='blue', alpha=0.6)
plt.scatter(df['temperatura'][df['fallo'] == 1], df['corriente'][df['fallo'] == 1], 
            c='red', alpha=0.6)
plt.title('Temperatura vs Corriente')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Corriente (A)')
plt.axhline(20, color='gray', linestyle='--', alpha=0.5)  # Límite superior corriente
plt.axhline(10, color='gray', linestyle='--', alpha=0.5)  # Límite inferior corriente
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)  # Umbral temperatura

# Presión vs Corriente
plt.subplot(1, 3, 3)
plt.scatter(df['presion'][df['fallo'] == 0], df['corriente'][df['fallo'] == 0], 
            c='blue', alpha=0.6)
plt.scatter(df['presion'][df['fallo'] == 1], df['corriente'][df['fallo'] == 1], 
            c='red', alpha=0.6)
plt.title('Presión vs Corriente')
plt.xlabel('Presión (psi)')
plt.ylabel('Corriente (A)')
plt.axhline(20, color='gray', linestyle='--', alpha=0.5)
plt.axhline(10, color='gray', linestyle='--', alpha=0.5)
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)  # Umbral presión

plt.tight_layout()
plt.show()

# Mostrar primeros 5 registros del array
print("\nEjemplo del array de máquinas:")
print(df.head().to_markdown(index=False))