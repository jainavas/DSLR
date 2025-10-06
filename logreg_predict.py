import pandas as pd
import numpy as np
import sys
import pickle

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def sigmoid(z):
    """
    Función sigmoide estable numéricamente
    """
    # Clip para evitar overflow
    z = np.clip(z, -500, 500)  # Limitar valores extremos
    return 1 / (1 + np.exp(-z))

def normalize_features(X, mins, maxs):
    """
    Normalizar usando los MISMOS mins/maxs del entrenamiento
    """
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Evitar división por cero
    
    X_norm = (X - mins) / ranges
    
    return X_norm

def predict(X, all_theta, houses):
    """
    Predecir la casa para cada estudiante
    """
    # ===== DEBUG START =====
    print(f"\n{'='*60}")
    print("DEBUG: Inside predict() function")
    print(f"{'='*60}")
    print(f"X shape: {X.shape}")
    print(f"Number of students: {X.shape[0]}")
    print(f"Number of features (including bias): {X.shape[1]}")
    print(f"Houses to predict: {houses}")
    
    # Verificar X
    print(f"\nX statistics:")
    print(f"  Contains NaN: {np.any(np.isnan(X))}")
    print(f"  Contains Inf: {np.any(np.isinf(X))}")
    print(f"  Min value: {np.min(X)}")
    print(f"  Max value: {np.max(X)}")
    print(f"  Mean value: {np.mean(X)}")
    
    # Verificar theta de cada casa
    print(f"\nTheta verification:")
    for house in houses:
        theta = all_theta[house]
        print(f"  {house}:")
        print(f"    Shape: {theta.shape}")
        print(f"    Contains NaN: {np.any(np.isnan(theta))}")
        print(f"    Contains Inf: {np.any(np.isinf(theta))}")
        print(f"    All zeros: {np.all(theta == 0)}")
        print(f"    First 3 values: {theta[:3]}")
    
    # Verificar dimensiones compatibles
    expected_theta_size = X.shape[1]  # Debe coincidir con número de features
    for house in houses:
        theta = all_theta[house]
        if theta.shape[0] != expected_theta_size:
            print(f"\n!!! ERROR: Dimension mismatch for {house}!")
            print(f"    X has {X.shape[1]} features but theta has {theta.shape[0]}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Starting predictions...")
    print(f"{'='*60}\n")
    # ===== DEBUG END =====
    
    m = X.shape[0]  # Número de estudiantes
    predictions = []  # Lista para guardar predicciones
    
    # DEBUG: Predecir primeros 5 estudiantes con detalle
    debug_limit = 5
    
    # Para cada estudiante
    for i in range(m):
        x_i = X[i, :]  # Features del estudiante i
        
        max_prob = -1  # Probabilidad máxima
        best_house = None  # Mejor casa
        
        # DEBUG para primeros estudiantes
        if i < debug_limit:
            print(f"\n--- Student {i} ---")
            print(f"Features (first 5): {x_i[:5]}")
        
        # Probar cada modelo (casa)
        house_probs = {}  # Para debug: guardar todas las probabilidades
        
        for house in houses:
            theta = all_theta[house]
            
            # Calcular z = x · theta
            z = x_i.dot(theta)  # Producto punto
            
            # Calcular probabilidad
            prob = sigmoid(z)
            
            house_probs[house] = prob
            
            # DEBUG
            if i < debug_limit:
                print(f"  {house}: z={z:.4f}, prob={prob:.4f}")
            
            # Actualizar mejor casa
            if prob > max_prob:
                max_prob = prob
                best_house = house
        
        # DEBUG
        if i < debug_limit:
            print(f"  PREDICTED: {best_house} (prob={max_prob:.4f})")
        
        # Verificar que se hizo una predicción válida
        if best_house is None:
            print(f"\n!!! ERROR: No prediction made for student {i}!")
            print(f"    Probabilities: {house_probs}")
            best_house = houses[0]  # Fallback: elegir primera casa
        
        predictions.append(best_house)
    
    # Verificar predicciones finales
    print(f"\n{'='*60}")
    print("Prediction summary:")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    
    # Contar predicciones por casa
    unique, counts = np.unique(predictions, return_counts=True)
    for house, count in zip(unique, counts):
        print(f"  {house}: {count} students ({count/len(predictions)*100:.1f}%)")
    
    # Verificar si hay None o NaN
    none_count = sum(1 for p in predictions if p is None)
    if none_count > 0:
        print(f"\n!!! WARNING: {none_count} predictions are None!")
    
    return np.array(predictions)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def predict_houses(test_file, model_file='weights.pkl'):
    """Función principal de predicción"""
    
    print(f"{'='*60}")
    print("LOGISTIC REGRESSION - PREDICTION")
    print(f"{'='*60}\n")
    
    # 1. CARGAR MODELO
    print("Step 1: Loading model...")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from '{model_file}'")
    except FileNotFoundError:
        print(f"✗ ERROR: Model file '{model_file}' not found!")
        print(f"   Run logreg_train.py first to create the model.")
        sys.exit(1)
    
    all_theta = model['theta']
    feature_columns = model['features']
    houses = model['houses']
    mins = model['mins']
    maxs = model['maxs']
    
    print(f"  Features: {len(feature_columns)}")
    print(f"  Houses: {list(houses)}")
    print(f"  Feature names: {feature_columns}\n")
    
    # 2. CARGAR DATOS DE TEST
    print("Step 2: Loading test dataset...")
    try:
        df = pd.read_csv(test_file)
        print(f"✓ Test data loaded: {len(df)} students")
    except FileNotFoundError:
        print(f"✗ ERROR: Test file '{test_file}' not found!")
        sys.exit(1)
    
    # Verificar que tiene columna Index
    if 'Index' not in df.columns:
        print("✗ ERROR: Test file must have 'Index' column!")
        sys.exit(1)
    
    original_indices = df['Index'].values
    print(f"  Index range: {original_indices[0]} to {original_indices[-1]}\n")
    
    # 3. VERIFICAR QUE FEATURES EXISTEN
    print("Step 3: Verifying features...")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"✗ ERROR: Missing features in test data:")
        for feat in missing_features:
            print(f"    - {feat}")
        sys.exit(1)
    
    print(f"✓ All features present in test data\n")
    
    # 4. EXTRAER FEATURES
    print("Step 4: Extracting features...")
    X = df[feature_columns].values
    print(f"✓ Features extracted: shape {X.shape}")
    
    # Verificar NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values found")
        print(f"  Filling NaN with column means...")
        
        # Rellenar NaN con media de cada columna
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]
        
        # Verificar que no quedan NaN
        remaining_nan = np.isnan(X).sum()
        if remaining_nan > 0:
            print(f"✗ ERROR: Still {remaining_nan} NaN after filling!")
            sys.exit(1)
        
        print(f"✓ NaN filled successfully")
    else:
        print(f"✓ No NaN values found")
    
    print()
    
    # 5. NORMALIZAR
    print("Step 5: Normalizing features...")
    print(f"  Using mins/maxs from training:")
    print(f"    Mins (first 3): {mins[:3]}")
    print(f"    Maxs (first 3): {maxs[:3]}")
    
    X_norm = normalize_features(X, mins, maxs)
    
    print(f"✓ Features normalized")
    print(f"  X_norm range: [{np.min(X_norm):.4f}, {np.max(X_norm):.4f}]")
    print(f"  X_norm mean: {np.mean(X_norm):.4f}")
    
    # Verificar normalización
    if np.any(np.isnan(X_norm)):
        print(f"✗ ERROR: NaN in normalized data!")
        sys.exit(1)
    if np.any(np.isinf(X_norm)):
        print(f"✗ ERROR: Inf in normalized data!")
        sys.exit(1)
    
    print()
    
    # 6. AÑADIR BIAS (columna de 1s)
    print("Step 6: Adding bias term...")
    ones = np.ones((X_norm.shape[0], 1))
    X_norm = np.hstack([ones, X_norm])
    print(f"✓ Bias added: new shape {X_norm.shape}")
    print()
    
    # 7. PREDECIR
    print("Step 7: Making predictions...")
    predictions = predict(X_norm, all_theta, houses)
    
    # 8. GUARDAR RESULTADOS
    print(f"\n{'='*60}")
    print("Step 8: Saving results...")
    print(f"{'='*60}\n")
    
    # Crear DataFrame
    output_df = pd.DataFrame({
        'Index': original_indices,
        'Hogwarts House': predictions
    })
    
    # Verificar que no hay None o NaN en predicciones
    if output_df['Hogwarts House'].isna().any():
        print("✗ ERROR: Some predictions are NaN!")
        print(output_df[output_df['Hogwarts House'].isna()])
        sys.exit(1)
    
    # Guardar
    output_df.to_csv('houses.csv', index=False)
    print(f"✓ Predictions saved to 'houses.csv'")
    print(f"  Total predictions: {len(predictions)}")
    
    # Mostrar primeras predicciones
    print(f"\nFirst 10 predictions:")
    print(output_df.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE!")
    print(f"{'='*60}\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py <dataset_test.csv>")
        sys.exit(1)
    
    predict_houses(sys.argv[1])