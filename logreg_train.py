import pandas as pd
import numpy as np
import sys
import pickle  # Para guardar/cargar objetos Python (como serializar structs en C)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def sigmoid(z):
    """
    Función sigmoide: convierte cualquier número en probabilidad [0, 1]
    σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))  # np.exp() = e^x, como exp() en <math.h>
    # np.exp funciona con arrays completos, no necesitas loop

def normalize_features(X):
    """
    Normalización: escalar features al rango [0, 1]
    Por qué: Si una feature tiene valores 0-100 y otra 0-1,
    la primera dominará el entrenamiento (pesos desbalanceados)
    
    Formula: x_norm = (x - min) / (max - min)
    """
    X_norm = np.zeros_like(X)  # Crear array de ceros del mismo tamaño (como calloc en C)
    mins = np.min(X, axis=0)   # Mínimo de cada columna (axis=0 = vertical)
    maxs = np.max(X, axis=0)   # Máximo de cada columna
    
    # Evitar división por cero si min == max
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Si rango es 0, poner 1 para no dividir por 0
    
    X_norm = (X - mins) / ranges  # Broadcasting: opera en todo el array a la vez
    
    return X_norm, mins, maxs  # Devolver mins/maxs para normalizar test data después

def compute_cost(X, y, theta):
    """
    Función de coste (loss function)
    Mide qué tan mal lo está haciendo el modelo
    
    J(θ) = -(1/m) Σ [y·log(h(x)) + (1-y)·log(1-h(x))]
    
    Cuanto menor, mejor (0 = perfecto)
    """
    m = len(y)  # Número de ejemplos (estudiantes)
    
    h = sigmoid(X.dot(theta))  # Predicción: h = σ(X·θ), dot() = producto matricial
    
    # Calcular coste
    # np.log() = logaritmo natural (ln), como log() en <math.h>
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Gradient Descent: algoritmo que ajusta los pesos iterativamente
    
    Parámetros:
    - X: matriz de features (m × n)
    - y: vector de labels (m × 1)
    - theta: pesos iniciales (n × 1)
    - alpha: learning rate (tamaño del paso)
    - num_iterations: cuántas veces iterar
    
    Retorna:
    - theta: pesos entrenados
    - costs: historial de costes (para ver si converge)
    """
    m = len(y)  # Número de ejemplos
    costs = []  # Lista para guardar historial de coste (como vector en C++)
    
    for i in range(num_iterations):  # for (int i = 0; i < num_iterations; i++)
        # 1. Calcular predicción
        h = sigmoid(X.dot(theta))  # h = σ(X·θ)
        
        # 2. Calcular error
        error = h - y  # Diferencia entre predicción y valor real
        
        # 3. Calcular gradiente (derivada parcial)
        gradient = (1/m) * X.T.dot(error)  # X.T = transpuesta de X
        # X.T.dot(error) = Σ(error · x) para cada feature
        
        # 4. Actualizar pesos
        theta = theta - alpha * gradient  # Descender por el gradiente
        
        # 5. Guardar coste cada cierto número de iteraciones (para monitorizar)
        if i % 100 == 0:  # Cada 100 iteraciones
            cost = compute_cost(X, y, theta)
            costs.append(cost)
            print(f"Iteration {i}: cost = {cost:.4f}")  # f-string = printf("%.4f", cost)
    
    return theta, costs

def train_one_vs_all(X, y, houses, alpha=0.01, num_iterations=1000):
    """
    One-vs-All: entrenar un modelo binario por cada casa
    
    Parámetros:
    - X: matriz de features normalizadas
    - y: vector con nombres de casas
    - houses: lista de casas únicas ['Gryffindor', 'Slytherin', ...]
    - alpha: learning rate
    - num_iterations: iteraciones de gradient descent
    
    Retorna:
    - all_theta: diccionario {casa: theta} con los pesos de cada modelo
    """
    n_features = X.shape[1]  # Número de columnas (features) - X.shape = (filas, columnas)
    all_theta = {}  # Diccionario vacío (std::map en C++)
    
    for house in houses:  # Para cada casa
        print(f"\nTraining model for {house}...")
        
        # Crear labels binarios: 1 si es esta casa, 0 si no
        y_binary = (y == house).astype(int)  # Convertir booleano a int (0 o 1)
        # Ejemplo: ['Gryff', 'Slyth', 'Gryff'] con house='Gryff' → [1, 0, 1]
        
        # Inicializar theta con ceros
        theta = np.zeros(n_features)  # Array de ceros (como calloc)
        
        # Entrenar modelo para esta casa
        theta, costs = gradient_descent(X, y_binary, theta, alpha, num_iterations)
        
        # Guardar theta de esta casa
        all_theta[house] = theta
    
    return all_theta

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def train(filename):
    """Función principal de entrenamiento"""
    
    # 1. CARGAR DATOS
    print("Loading dataset...")
    df = pd.read_csv(filename)
    
    # 2. SELECCIONAR FEATURES
    # Aquí debes elegir las features que viste en el pair_plot
    # Por ahora usamos todas las numéricas (tú deberías filtrar)
    feature_columns = [
        'Astronomy', 'Herbology', 'Divination', 'Muggle Studies',
        'Ancient Runes', 'History of Magic', 'Transfiguration',
        'Potions', 'Care of Magical Creatures', 'Charms',
        'Flying', 'Defense Against the Dark Arts'
    ]  # Lista de strings (char *features[] en C)
    
    # 3. PREPARAR DATOS
    print("Preparing data...")
    
    # Filtrar filas con datos completos (eliminar NaN)
    df_clean = df[feature_columns + ['Hogwarts House']].dropna()
    
    # Separar X (features) e y (labels)
    X = df_clean[feature_columns].values  # .values = convertir DataFrame a numpy array
    y = df_clean['Hogwarts House'].values  # Array de strings
    
    # 4. NORMALIZAR FEATURES
    print("Normalizing features...")
    X_norm, mins, maxs = normalize_features(X)
    
    # Añadir columna de 1s al inicio (bias term / intercept)
    # Esto permite que el modelo tenga un término independiente: θ₀ + θ₁x₁ + ...
    ones = np.ones((X_norm.shape[0], 1))  # Columna de 1s (m × 1)
    X_norm = np.hstack([ones, X_norm])  # hstack = concatenar horizontalmente
    # Ahora X_norm tiene shape (m, n+1)
    
    # 5. OBTENER CASAS ÚNICAS
    houses = np.unique(y)  # Array con casas únicas (como hacer set en C++)
    print(f"Houses: {houses}")
    
    # 6. ENTRENAR MODELOS
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    all_theta = train_one_vs_all(
        X_norm, y, houses,
        alpha=0.01,        # Learning rate (prueba valores: 0.001, 0.01, 0.1)
        num_iterations=2000  # Número de iteraciones (1000-5000 típico)
    )
    
    # 7. GUARDAR MODELO
    print("\nSaving model...")
    
    model = {
        'theta': all_theta,           # Pesos entrenados
        'features': feature_columns,  # Qué features usamos
        'houses': houses,             # Orden de las casas
        'mins': mins,                 # Para normalizar test data
        'maxs': maxs
    }  # Diccionario con toda la info necesaria
    
    print("\n=== MODEL VERIFICATION ===")
    for house, theta in all_theta.items():
        print(f"{house}: theta shape = {theta.shape}, theta[0:3] = {theta[0:3]}")
        # Verificar que no son todos ceros o NaN
        if np.all(theta == 0):
            print(f"  WARNING: All zeros for {house}!")
        if np.any(np.isnan(theta)):
            print(f"  WARNING: NaN detected in {house}!")

    # Guardar en archivo usando pickle (serialización)
    with open('weights.pkl', 'wb') as f:  # 'wb' = write binary
        pickle.dump(model, f)  # pickle.dump = serializar objeto
    
    print("Model saved to 'weights.pkl'")
    print("\nTraining complete!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)
    
    train(sys.argv[1])