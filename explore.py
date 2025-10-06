import pandas as pd

# Cargar el CSV
df = pd.read_csv('dataset_train.csv')

# Ver las primeras filas
print(df.head())

# Ver los nombres de las columnas
print(df.columns)

# Ver información general
print(df.info())

# Ver estadísticas (ESTO NO LO PUEDES USAR EN TU DESCRIBE.PY)
print(df.describe())