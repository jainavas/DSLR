# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:26:14 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:27:48 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_histogram(filename):
    # Leer CSV
    df = pd.read_csv(filename)
    
    # Obtener nombres de materias (columnas numéricas, excluyendo Index, Birthday...)
    courses = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'Index']
    # List comprehension: filtra columnas que sean números
    
    # Crear figura con subplots (ventanas de gráficos)
    # Como crear múltiples ventanas en una GUI
    n_courses = len(courses)  # Número de materias
    cols = 4  # 4 columnas de gráficos
    rows = (n_courses + cols - 1) // cols  # Cálculo de filas necesarias (ceiling division)
    
    # plt.subplots crea una cuadrícula de gráficos
    # figsize=(ancho, alto) en pulgadas
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 7))
    axes = axes.flatten()  # Convertir matriz 2D en array 1D para iterar fácil
    
    # Definir colores para cada casa
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }
    
    # Para cada materia
    for idx, course in enumerate(courses):
        ax = axes[idx]
        
        # Para cada casa, dibujar histograma
        for house in colors.keys():  # Iterar sobre las claves del diccionario
            # Filtrar estudiantes de esta casa
            house_data = df[df['Hogwarts House'] == house][course]  # SQL: SELECT course WHERE house = 'X'
            # df['Hogwarts House'] == house devuelve array booleano
            # df[boolean_array] filtra filas donde es True
            
            # Eliminar NaN (valores faltantes)
            house_data = house_data.dropna()
            
            # Dibujar histograma
            # alpha=0.5 = transparencia (para que se vean superpuestos)
            # bins=20 = número de "barras" en el histograma
            ax.hist(house_data, bins=20, alpha=0.5, label=house, color=colors[house])
        
        ax.set_title(course)  # Título del gráfico (como poner texto en ventana)
        ax.set_xlabel('Score')  # Etiqueta eje X
        ax.set_ylabel('Frequency')  # Etiqueta eje Y
        ax.legend()  # Mostrar leyenda con los colores de cada casa
    
    # Ocultar subplots vacíos si sobran
    for idx in range(n_courses, len(axes)):  # for restante hasta llenar la cuadrícula
        axes[idx].axis('off')  # Ocultar ejes
    
    plt.tight_layout(pad = 2.7)  # Ajustar espaciado automáticamente
    plt.subplots_adjust(hspace=0.4)
    plt.show()  # Mostrar ventana (como un bucle de eventos en GUI)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>")
        sys.exit(1)
    
    plot_histogram(sys.argv[1])