import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Nueva librería para 3D

# Configuración inicial
np.random.seed(42)
num_maquinas = 300  # Aumentamos a 300 máquinas

# Generar datos sintéticos
data = {
    'temperatura': np.concatenate([
        np.random.normal(50, 10, 150),  # Normales
        np.random.normal(85, 5, 75),     # Altas
        np.random.normal(90, 8, 75)      # Muy altas
    ]),
    'presion': np.concatenate([
        np.random.normal(100, 5, 150),   # Óptimas
        np.random.normal(70, 8, 75),     # Bajas
        np.random.normal(60, 10, 75)     # Muy bajas
    ]),
    'corriente': np.concatenate([
        np.random.normal(15, 2, 150),    # Normales
        np.random.normal(25, 3, 75),     # Altas
        np.random.normal(5, 1.5, 75)     # Bajas
    ])
}

df = pd.DataFrame(data)

# Crear etiquetas (1 = fallo próximo, 0 = normal)
df['fallo'] = 0
df.loc[
    ((df['temperatura'] > 80) & 
    (df['presion'] < 80)) |
    ((df['temperatura'] > 80) & 
    ((df['corriente'] > 20) | (df['corriente'] < 10))) |
    ((df['presion'] < 80) & 
    ((df['corriente'] > 20) | (df['corriente'] < 10))),
    'fallo'
] = 1

# Guardar en JSON
df.to_json("maquinas.json", orient="records", indent=4)

# Gráficos 2D (con más datos)
plt.figure(figsize=(15, 5))
# ... (los mismos gráficos 2D que antes, ahora con más puntos)
plt.tight_layout()
plt.show()

# Nueva Gráfica 3D (combinando las 3 métricas)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['temperatura'],
    df['presion'],
    df['corriente'],
    c=df['fallo'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='w'
)

ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Presión (psi)')
ax.set_zlabel('Corriente (A)')
ax.set_title('Patrones de Fallo: 3 Métricas Combinadas')

# Leyenda personalizada
legend = ax.legend(*scatter.legend_elements(), title="Fallo")
ax.add_artist(legend)

# Líneas de umbrales
ax.plot([80, 80], [0, 100], [0, 30], color='red', linestyle='--', alpha=0.5)  # Temp
ax.plot([0, 100], [80, 80], [0, 30], color='blue', linestyle='--', alpha=0.5)  # Presión
ax.plot([0, 100], [0, 100], [10, 10], color='green', linestyle='--', alpha=0.5)  # Corriente baja
ax.plot([0, 100], [0, 100], [20, 20], color='orange', linestyle='--', alpha=0.5)  # Corriente alta

plt.show()