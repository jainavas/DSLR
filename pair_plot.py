# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:23:32 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:24:19 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plot_pair_plot(filename):
    df = pd.read_csv(filename)
    
    # Seleccionar solo columnas numéricas y la columna de casa
    courses = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'Index']
    
    # Limitar a las características más importantes (opcional, si son muchas)
    # Para no tener 100 gráficos
    if len(courses) > 8:  # Si hay más de 8 materias
        # Seleccionar las primeras 8 (o usa otro criterio)
        courses = courses[:8]  # Array slicing [inicio:fin] en Python
    
    # Crear DataFrame solo con las columnas relevantes
    plot_df = df[courses + ['Hogwarts House']].dropna()  # Concatenar listas + operator
    
    # Crear pair plot
    # hue='Hogwarts House' = colorear por casa
    # diag_kind='hist' = en la diagonal poner histogramas (no scatter plots)
    # palette = mapa de colores
    g = sns.pairplot(
        plot_df,  # DataFrame a graficar
        hue='Hogwarts House',  # Columna para colorear
        diag_kind='hist',  # Tipo de gráfico en diagonal
        palette={  # Diccionario de colores
            'Gryffindor': 'red',
            'Slytherin': 'green',
            'Ravenclaw': 'blue',
            'Hufflepuff': 'purple'
        },
        plot_kws={'alpha': 0.6, 's': 30},  # Kwargs = argumentos opcionales (transparencia, tamaño)
        diag_kws={'alpha': 0.7}  # Opciones para diagonales
    )
    g.figure.subplots_adjust(
        hspace=0.35,       # Espacio vertical
        wspace=0.35,       # Espacio horizontal
        top=0.95,
        bottom=0.05,
        left=0.05,
        right=0.98
    )
    
    plt.suptitle('Pair Plot - Feature Relationships', y=1.01)
    plt.tight_layout(pad = 3.0)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>")
        sys.exit(1)
    
    plot_pair_plot(sys.argv[1])