# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:26:04 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:26:07 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np  # Operaciones matemáticas con arrays

def plot_scatter(filename):
    df = pd.read_csv(filename)
    
    # Obtener solo columnas numéricas
    courses = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'Index']
    
    # Calcular matriz de correlación
    # Esto calcula qué tan relacionadas están todas las características entre sí
    # Valores cercanos a 1 = muy correlacionadas
    # Valores cercanos a 0 = no correlacionadas
    # Valores cercanos a -1 = correlación inversa
    numeric_df = df[courses].dropna()  # Filtrar solo columnas numéricas sin NaN
    
    # Encontrar el par con mayor correlación (excluyendo diagonal)
    max_corr = 0
    best_pair = (None, None)
    
    # Comparar todas las combinaciones de materias
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            course1 = courses[i]
            course2 = courses[j]
            
            # Obtener datos de ambas materias
            data1 = df[course1].dropna()  # Filtrar NaN
            data2 = df[course2].dropna()
            
            # Encontrar índices comunes (estudiantes que tienen nota en ambas)
            common_indices = data1.index.intersection(data2.index)  # Intersección de conjuntos
            
            if len(common_indices) < 2:  # Si hay menos de 2 datos, saltar
                continue
            
            # Calcular correlación de Pearson manualmente
            # r = Σ((x - mean_x)(y - mean_y)) / sqrt(Σ(x - mean_x)² * Σ(y - mean_y)²)
            x = data1[common_indices].values
            y = data2[common_indices].values
            
            mean_x = np.mean(x)  # Media de x
            mean_y = np.mean(y)  # Media de y
            
            # Numerador: covarianza
            numerator = np.sum((x - mean_x) * (y - mean_y))  # Suma de productos
            
            # Denominador: producto de desviaciones estándar
            denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
            
            if denominator == 0:  # Evitar división por cero
                continue
            
            correlation = abs(numerator / denominator)  # Valor absoluto (nos interesa magnitud)
            
            if correlation > max_corr:  # Si encontramos mayor correlación
                max_corr = correlation
                best_pair = (course1, course2)
    
    # Dibujar scatter plot del par más correlacionado
    course1, course2 = best_pair
    
    # Colores por casa
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'purple'
    }
    
    plt.figure(figsize=(10, 8))  # Crear ventana gráfica
    
    # Para cada casa, dibujar puntos
    for house in colors.keys():
        house_df = df[df['Hogwarts House'] == house]  # Filtrar por casa
        
        x = house_df[course1].dropna()  # Datos eje X
        y = house_df[course2].dropna()  # Datos eje Y
        
        # Encontrar índices comunes
        common = x.index.intersection(y.index)
        
        # Dibujar puntos (scatter)
        # s=50 = tamaño de puntos
        # alpha=0.6 = transparencia
        plt.scatter(x[common], y[common], label=house, color=colors[house], s=50, alpha=0.6)
    
    plt.xlabel(course1)  # Etiqueta eje X
    plt.ylabel(course2)  # Etiqueta eje Y
    plt.title(f'Most correlated features (r={max_corr:.3f})')  # f-string = sprintf en C
    plt.legend()  # Mostrar leyenda
    plt.grid(True, alpha=0.3)  # Mostrar cuadrícula de fondo
    plt.tight_layout()
    plt.show()  # Mostrar ventana

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)
    
    plot_scatter(sys.argv[1])