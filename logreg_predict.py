# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_predict.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:23:27 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:23:30 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys
import pickle

def sigmoid(z):
    """Función sigmoide"""
    z = np.clip(z, -500, 500)  # Evitar overflow
    return 1 / (1 + np.exp(-z))

def normalize_features(X, mins, maxs):
    """Normalizar features con mins/maxs del entrenamiento"""
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    X_norm = (X - mins) / ranges
    return X_norm

def predict(X, all_theta, houses):
    """Predecir la casa para cada estudiante"""
    m = X.shape[0]  # Número de estudiantes
    predictions = []
    
    for i in range(m):
        x_i = X[i, :]  # Features del estudiante i
        
        max_prob = -1
        best_house = None
        
        # Probar cada modelo (casa)
        for house in houses:
            theta = all_theta[house]
            z = x_i.dot(theta)
            prob = sigmoid(z)
            
            if prob > max_prob:
                max_prob = prob
                best_house = house
        
        predictions.append(best_house)
    
    return np.array(predictions)

def predict_houses(test_file, model_file='weights.pkl'):
    """Función principal de predicción"""
    
    # 1. Cargar modelo
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found.")
        print("Run logreg_train.py first to create the model.")
        sys.exit(1)
    
    all_theta = model['theta']
    feature_columns = model['features']
    houses = model['houses']
    mins = model['mins']
    maxs = model['maxs']
    
    # 2. Cargar datos de test
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found.")
        sys.exit(1)
    
    # Verificar columna Index
    if 'Index' not in df.columns:
        print("Error: Test file must have 'Index' column.")
        sys.exit(1)
    
    original_indices = df['Index'].values
    
    # 3. Verificar features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Error: Missing features in test data: {missing_features}")
        sys.exit(1)
    
    # 4. Extraer features
    X = df[feature_columns].values
    
    # Rellenar NaN con media de columna
    if np.isnan(X).sum() > 0:
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]
    
    # 5. Normalizar
    X_norm = normalize_features(X, mins, maxs)
    
    # Añadir columna de 1s (bias)
    ones = np.ones((X_norm.shape[0], 1))
    X_norm = np.hstack([ones, X_norm])
    
    # 6. Predecir
    predictions = predict(X_norm, all_theta, houses)
    
    # 7. Guardar resultados
    output_df = pd.DataFrame({
        'Index': original_indices,
        'Hogwarts House': predictions
    })
    
    output_df.to_csv('houses.csv', index=False)
    print(f"Predictions saved to 'houses.csv'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <dataset_test.csv>")
        sys.exit(1)
    
    predict_houses(sys.argv[1])