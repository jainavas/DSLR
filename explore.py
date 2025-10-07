# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    explore.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:29:16 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:29:20 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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