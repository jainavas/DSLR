import sys
import pandas as pd
import numpy as np

def ft_count(data):
    """Cuenta valores no-NaN"""
    count = 0
    for value in data:
        if not pd.isna(value):  # pd.isna() es como comprobar NULL
            count += 1
    return count

def ft_mean(data):
    """Calcula la media"""
    # Filtrar valores NaN
    clean_data = [x for x in data if not pd.isna(x)]
    
    if len(clean_data) == 0:
        return float('nan')
    
    total = sum(clean_data)
    return total / len(clean_data)

def ft_std(data):
    """Calcula la desviación estándar"""
    clean_data = [x for x in data if not pd.isna(x)]
    
    if len(clean_data) < 2:
        return float('nan')
    
    mean = ft_mean(data)
    
    # Suma de (valor - media)²
    variance_sum = sum((x - mean) ** 2 for x in clean_data)
    
    # Dividir por n-1 (sample standard deviation)
    variance = variance_sum / (len(clean_data) - 1)
    
    # Raíz cuadrada
    std = variance ** 0.5
    
    return std

def ft_min(data):
    """Encuentra el mínimo"""
    clean_data = [x for x in data if not pd.isna(x)]
    
    if len(clean_data) == 0:
        return float('nan')
    
    minimum = clean_data[0]
    for value in clean_data:
        if value < minimum:
            minimum = value
    
    return minimum

def ft_max(data):
    """Encuentra el máximo"""
    clean_data = [x for x in data if not pd.isna(x)]
    
    if len(clean_data) == 0:
        return float('nan')
    
    maximum = clean_data[0]
    for value in clean_data:
        if value > maximum:
            maximum = value
    
    return maximum

def ft_percentile(data, percentile):
    """
    Calcula un percentil
    percentile: valor entre 0 y 1 (0.25 para 25%, 0.5 para 50%, etc.)
    """
    clean_data = [x for x in data if not pd.isna(x)]
    
    if len(clean_data) == 0:
        return float('nan')
    
    # Ordenar los datos
    sorted_data = sorted(clean_data)
    n = len(sorted_data)
    
    # Calcular la posición
    position = percentile * (n - 1)
    
    # Si la posición es entera, devolver ese valor
    if position.is_integer():
        return sorted_data[int(position)]
    
    # Si no, interpolar entre los dos valores cercanos
    lower_index = int(position)
    upper_index = lower_index + 1
    fraction = position - lower_index
    
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    
    # Interpolación lineal
    result = lower_value + fraction * (upper_value - lower_value)
    
    return result

def describe(filename):
    """Función principal"""
    # Leer el CSV
    df = pd.read_csv(filename)
    
    # Seleccionar solo columnas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Crear diccionario con estadísticas
    stats = {
        'Count': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'Max': []
    }
    
    # Calcular estadísticas para cada columna
    for col in numeric_columns:
        data = df[col].tolist()
        
        stats['Count'].append(ft_count(data))
        stats['Mean'].append(ft_mean(data))
        stats['Std'].append(ft_std(data))
        stats['Min'].append(ft_min(data))
        stats['25%'].append(ft_percentile(data, 0.25))
        stats['50%'].append(ft_percentile(data, 0.50))
        stats['75%'].append(ft_percentile(data, 0.75))
        stats['Max'].append(ft_max(data))
    
    # Crear DataFrame con los resultados
    result_df = pd.DataFrame(stats, index=numeric_columns).T
    
    # Imprimir con formato
    print(result_df.to_string(float_format='%.6f'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)
    
    describe(sys.argv[1])