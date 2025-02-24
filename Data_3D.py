import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar datos
df = pd.read_json("maquinas.json")

# Configurar gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D
scatter = ax.scatter(
    df['temperatura'],
    df['presion'],
    df['corriente'],
    c=df['fallo'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='w'
)

# Personalizar ejes
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Presión (psi)')
ax.set_zlabel('Corriente (A)')
ax.set_title('Patrones de Fallo: 3 Métricas Combinadas')

# Añadir leyenda
legend = ax.legend(*scatter.legend_elements(), title="Fallo")
ax.add_artist(legend)

# Líneas de umbrales
ax.plot([80, 80], [0, 100], [0, 30], color='red', linestyle='--', alpha=0.5)
ax.plot([0, 100], [80, 80], [0, 30], color='blue', linestyle='--', alpha=0.5)
ax.plot([0, 100], [0, 100], [10, 10], color='green', linestyle='--', alpha=0.5)
ax.plot([0, 100], [0, 100], [20, 20], color='orange', linestyle='--', alpha=0.5)

plt.show()