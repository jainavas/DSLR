# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    evaluate.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jainavas <jainavas@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/10/07 18:29:14 by jainavas          #+#    #+#              #
#    Updated: 2025/10/07 18:29:15 by jainavas         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import sys
import numpy as np

def evaluate(predictions_file, truth_file):
    """
    Comparar predicciones con valores reales
    Calcular accuracy
    """
    
    print("="*60)
    print("EVALUATION")
    print("="*60 + "\n")
    
    # Cargar predicciones
    print("Loading predictions...")
    pred_df = pd.read_csv(predictions_file)
    print(f"✓ Loaded {len(pred_df)} predictions")
    print(f"  Columns: {list(pred_df.columns)}")
    print(f"  Sample:\n{pred_df.head()}\n")
    
    # Cargar valores reales
    print("Loading ground truth...")
    truth_df = pd.read_csv(truth_file)
    print(f"✓ Loaded {len(truth_df)} truth values")
    
    # Verificar si tiene columna Hogwarts House
    if 'Hogwarts House' not in truth_df.columns:
        print("✗ ERROR: Test file doesn't have 'Hogwarts House' column!")
        print(f"  Available columns: {list(truth_df.columns)}")
        print("\n  This test file doesn't contain labels.")
        print("  Cannot evaluate accuracy.")
        sys.exit(1)
    
    print(f"  Columns: {list(truth_df.columns)}")
    print(f"  Sample:\n{truth_df[['Index', 'Hogwarts House']].head()}\n")
    
    # Verificar longitudes
    if len(pred_df) != len(truth_df):
        print(f"✗ ERROR: Different number of rows!")
        print(f"  Predictions: {len(pred_df)}")
        print(f"  Truth: {len(truth_df)}")
        sys.exit(1)
    
    # Limpiar espacios en blanco
    print("Cleaning data...")
    pred_df['Hogwarts House'] = pred_df['Hogwarts House'].str.strip()  # Quitar espacios
    truth_df['Hogwarts House'] = truth_df['Hogwarts House'].str.strip()
    print("✓ Whitespace removed\n")
    
    # Verificar valores únicos
    print("Checking unique values...")
    pred_houses = set(pred_df['Hogwarts House'].unique())
    truth_houses = set(truth_df['Hogwarts House'].unique())
    
    print(f"  Predicted houses: {pred_houses}")
    print(f"  Truth houses: {truth_houses}")
    
    # Verificar si hay diferencias
    only_in_pred = pred_houses - truth_houses
    only_in_truth = truth_houses - pred_houses
    
    if only_in_pred:
        print(f"  ⚠ Houses only in predictions: {only_in_pred}")
    if only_in_truth:
        print(f"  ⚠ Houses only in truth: {only_in_truth}")
    
    print()
    
    # IMPORTANTE: Ordenar por Index para comparar correctamente
    print("Sorting by Index...")
    pred_df = pred_df.sort_values('Index').reset_index(drop=True)  # Ordenar por Index
    truth_df = truth_df.sort_values('Index').reset_index(drop=True)
    print("✓ Both dataframes sorted by Index\n")
    
    # Verificar que los índices coinciden
    if not np.array_equal(pred_df['Index'].values, truth_df['Index'].values):
        print("⚠ WARNING: Index columns don't match exactly!")
        print("  Merging on Index column...")
        
        # Merge basado en Index
        merged = pred_df.merge(
            truth_df[['Index', 'Hogwarts House']], 
            on='Index', 
            suffixes=('_pred', '_truth')
        )
        
        predictions = merged['Hogwarts House_pred'].values
        truth = merged['Hogwarts House_truth'].values
        
        print(f"✓ Merged successfully: {len(merged)} rows\n")
    else:
        print("✓ Index columns match\n")
        predictions = pred_df['Hogwarts House'].values
        truth = truth_df['Hogwarts House'].values
    
    # Mostrar primeras comparaciones
    print("First 10 comparisons:")
    print(f"{'Index':<8} {'Predicted':<15} {'Truth':<15} {'Correct':<10}")
    print("-" * 50)
    for i in range(min(10, len(predictions))):
        idx = pred_df['Index'].iloc[i] if i < len(pred_df) else i
        correct = "✓" if predictions[i] == truth[i] else "✗"
        print(f"{idx:<8} {predictions[i]:<15} {truth[i]:<15} {correct:<10}")
    print()
    
    # Comparar predicciones con verdad
    print("="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    correct = (predictions == truth).sum()  # Contar aciertos
    total = len(predictions)
    accuracy = correct / total
    
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Meta del proyecto: ≥ 98%
    if accuracy >= 0.98:
        print("\n✓ Sorting Hat quality achieved! (≥98%)")
    else:
        needed = 0.98 - accuracy
        print(f"\n✗ Below Sorting Hat quality.")
        print(f"  Need {needed:.4f} ({needed*100:.2f}%) more accuracy")
    
    # Confusion matrix simplificada
    print("\n" + "="*60)
    print("Performance by house:")
    print("="*60)
    
    houses_list = sorted(set(truth))  # Lista de casas únicas ordenadas
    
    for house in houses_list:
        mask = truth == house  # Estudiantes de esta casa
        house_correct = (predictions[mask] == truth[mask]).sum()
        house_total = mask.sum()
        house_acc = house_correct / house_total if house_total > 0 else 0
        
        print(f"  {house:<15} {house_correct:>3}/{house_total:<3} ({house_acc*100:>5.1f}%)")
    
    # Mostrar errores más comunes
    print("\n" + "="*60)
    print("Most common errors:")
    print("="*60)
    
    errors = predictions != truth  # Booleano: donde hay errores
    if errors.sum() > 0:
        error_pred = predictions[errors]
        error_truth = truth[errors]
        
        # Contar pares (predicción incorrecta, valor real)
        error_pairs = list(zip(error_pred, error_truth))
        unique_pairs, counts = np.unique(error_pairs, return_counts=True, axis=0)
        
        # Ordenar por frecuencia
        sorted_indices = np.argsort(counts)[::-1]  # Descendente
        
        print(f"{'Predicted as':<15} {'Actually was':<15} {'Count':<10}")
        print("-" * 42)
        for i in sorted_indices[:5]:  # Top 5 errores
            pred_house, true_house = unique_pairs[i]
            count = counts[i]
            print(f"{pred_house:<15} {true_house:<15} {count:<10}")
    else:
        print("  No errors! Perfect predictions!")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py houses.csv dataset_test.csv")
        sys.exit(1)
    
    evaluate(sys.argv[1], sys.argv[2])