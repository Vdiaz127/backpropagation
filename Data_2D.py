import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_json("maquinas.json")

# Configurar gráficos
plt.figure(figsize=(15, 5))

# Gráfico 1: Temperatura vs Presión
plt.subplot(1, 3, 1)
plt.scatter(df['temperatura'][df['fallo'] == 0], df['presion'][df['fallo'] == 0],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(df['temperatura'][df['fallo'] == 1], df['presion'][df['fallo'] == 1],
            c='red', label='Fallo próximo', alpha=0.6)
plt.title('Temperatura vs Presión')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Presión (psi)')
plt.axhline(80, color='gray', linestyle='--', alpha=0.5)
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)
plt.legend()

# Gráfico 2: Temperatura vs Corriente
plt.subplot(1, 3, 2)
plt.scatter(df['temperatura'][df['fallo'] == 0], df['corriente'][df['fallo'] == 0],
            c='blue', alpha=0.6)
plt.scatter(df['temperatura'][df['fallo'] == 1], df['corriente'][df['fallo'] == 1],
            c='red', alpha=0.6)
plt.title('Temperatura vs Corriente')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Corriente (A)')
plt.axhline(20, color='gray', linestyle='--', alpha=0.5)
plt.axhline(10, color='gray', linestyle='--', alpha=0.5)
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)

# Gráfico 3: Presión vs Corriente
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
plt.axvline(80, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()