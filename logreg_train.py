import pandas as pd
import numpy as np
import sys
import pickle

def sigmoid(z):
    """Función sigmoide"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def normalize_features(X):
    """Normalizar features al rango [0, 1]"""
    X_norm = np.zeros_like(X)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    X_norm = (X - mins) / ranges
    
    return X_norm, mins, maxs

def compute_cost(X, y, theta):
    """Función de coste (loss function)"""
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # Evitar log(0)
    h = np.clip(h, 1e-10, 1 - 1e-10)
    
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    """Gradient Descent para ajustar los pesos"""
    m = len(y)
    costs = []
    
    for i in range(num_iterations):
        # Predicción
        h = sigmoid(X.dot(theta))
        
        # Error
        error = h - y
        
        # Gradiente
        gradient = (1/m) * X.T.dot(error)
        
        # Actualizar pesos
        theta = theta - alpha * gradient
        
        # Guardar coste cada 100 iteraciones
        if i % 100 == 0:
            cost = compute_cost(X, y, theta)
            costs.append(cost)
    
    return theta, costs

def train_one_vs_all(X, y, houses, alpha=0.01, num_iterations=2000):
    """Entrenar un modelo binario por cada casa (One-vs-All)"""
    n_features = X.shape[1]
    all_theta = {}
    
    for house in houses:
        print(f"Training model for {house}...")
        
        # Crear labels binarios: 1 si es esta casa, 0 si no
        y_binary = (y == house).astype(int)
        
        # Inicializar theta con ceros
        theta = np.zeros(n_features)
        
        # Entrenar
        theta, costs = gradient_descent(X, y_binary, theta, alpha, num_iterations)
        
        # Guardar theta
        all_theta[house] = theta
    
    return all_theta

def train(filename):
    """Función principal de entrenamiento"""
    
    print("Loading dataset...")
    df = pd.read_csv(filename)
    
    # Seleccionar features
    feature_columns = [
        'Astronomy', 'Herbology', 'Divination', 'Muggle Studies',
        'Ancient Runes', 'History of Magic', 'Transfiguration',
        'Potions', 'Care of Magical Creatures', 'Charms',
        'Flying', 'Defense Against the Dark Arts'
    ]
    
    # Preparar datos
    print("Preparing data...")
    df_clean = df[feature_columns + ['Hogwarts House']].dropna()
    
    X = df_clean[feature_columns].values
    y = df_clean['Hogwarts House'].values
    
    # Normalizar
    print("Normalizing features...")
    X_norm, mins, maxs = normalize_features(X)
    
    # Añadir columna de 1s (bias)
    ones = np.ones((X_norm.shape[0], 1))
    X_norm = np.hstack([ones, X_norm])
    
    # Obtener casas únicas
    houses = np.unique(y)
    print(f"Houses: {houses}")
    
    # Entrenar modelos
    print("\nStarting training...")
    all_theta = train_one_vs_all(
        X_norm, y, houses,
        alpha=0.01,
        num_iterations=2000
    )
    
    # Guardar modelo
    print("\nSaving model...")
    model = {
        'theta': all_theta,
        'features': feature_columns,
        'houses': houses,
        'mins': mins,
        'maxs': maxs
    }
    
    with open('weights.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to 'weights.pkl'")
    print("Training complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)
    
    train(sys.argv[1])