import numpy as np
import pandas as pd

# Configuración inicial
np.random.seed(42)
num_maquinas = 300

# Generar datos sintéticos
data = {
    'temperatura': np.concatenate([
        np.random.normal(50, 10, 150),
        np.random.normal(85, 5, 75),
        np.random.normal(90, 8, 75)
    ]),
    'presion': np.concatenate([
        np.random.normal(100, 5, 150),
        np.random.normal(70, 8, 75),
        np.random.normal(60, 10, 75)
    ]),
    'corriente': np.concatenate([
        np.random.normal(15, 2, 150),
        np.random.normal(25, 3, 75),
        np.random.normal(5, 1.5, 75)
    ])
}

df = pd.DataFrame(data)

# Crear etiquetas de fallo
df['fallo'] = 0
df.loc[
    ((df['temperatura'] > 80) & (df['presion'] < 80)) |
    ((df['temperatura'] > 80) & ((df['corriente'] > 20) | (df['corriente'] < 10))) |
    ((df['presion'] < 80) & ((df['corriente'] > 20) | (df['corriente'] < 10))),
    'fallo'
] = 1

# Guardar en JSON
df.to_json("maquinas.json", orient="records", indent=4)
print("¡JSON guardado correctamente como 'maquinas.json'!")

# Mostrar ejemplo de datos
print("\nEjemplo del dataset:")
print(df.head().to_markdown(index=False))