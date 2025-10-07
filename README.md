# DSLR - Data Science × Logistic Regression

Implementation of a logistic regression classifier to sort Hogwarts students into houses.

## Project Structure

```
dslr/
├── dataset_train.csv           # Training data
├── dataset_test.csv            # Test data (without labels)
├── describe.py                 # Statistical analysis
├── histogram.py                # Feature distribution visualization
├── scatter_plot.py             # Feature correlation analysis
├── pair_plot.py                # Scatter plot matrix
├── logreg_train.py            # Model training
├── logreg_predict.py          # House prediction
├── weights.pkl                 # Trained model (generated)
└── houses.csv                  # Predictions (generated)
```

## Installation

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install pandas numpy matplotlib seaborn
```

## Usage

### 1. Data Analysis

Displays statistical metrics (count, mean, std, min, quartiles, max) for all numerical features.

```bash
python describe.py dataset_train.csv
```

**Note:** No built-in functions like `pandas.describe()` or `numpy.mean()` are used. All statistics are calculated manually.

### 2. Data Visualization

#### Histogram
Find which course has a homogeneous score distribution across all four houses.

```bash
python histogram.py dataset_train.csv
```

#### Scatter Plot
Identify the two most similar/correlated features.

```bash
python scatter_plot.py dataset_train.csv
```

#### Pair Plot
Visualize all feature relationships to select the best features for the model.

```bash
python pair_plot.py dataset_train.csv
```

### 3. Training

Train the logistic regression model using gradient descent.

```bash
python logreg_train.py dataset_train.csv
```

This generates `weights.pkl` containing:
- Trained weights (theta) for each house
- Feature names used
- Normalization parameters (min/max values)

**Parameters:**
- Learning rate (alpha): `0.01`
- Iterations: `2000`
- Algorithm: Batch Gradient Descent
- Strategy: One-vs-All (4 binary classifiers)

### 4. Prediction

Generate predictions for the test dataset.

```bash
python logreg_predict.py dataset_test.csv
```

This generates `houses.csv` with predictions in the format:
```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
...
```

## Implementation Details

### Statistical Functions (describe.py)

All statistical functions are implemented from scratch:

- **Count**: Number of non-NaN values
- **Mean**: Sum of values divided by count
- **Std**: Standard deviation using Bessel's correction (n-1)
- **Min/Max**: Minimum and maximum values
- **Percentiles (25%, 50%, 75%)**: Calculated using linear interpolation

### Logistic Regression

#### Mathematical Foundation

**Hypothesis function:**
```
h(x) = σ(θᵀx)
where σ(z) = 1 / (1 + e⁻ᶻ)
```

**Cost function:**
```
J(θ) = -(1/m) Σ [y·log(h(x)) + (1-y)·log(1-h(x))]
```

**Gradient:**
```
∂J/∂θ = (1/m) Xᵀ(h(x) - y)
```

**Update rule:**
```
θ := θ - α·∇J(θ)
```

#### Key Features

1. **Feature Normalization**: All features are scaled to [0, 1] range to ensure balanced gradient descent convergence.

2. **Bias Term**: A column of ones is added to X to allow the decision boundary to not pass through the origin.

3. **One-vs-All Strategy**: Since we have 4 houses, we train 4 binary classifiers:
   - Gryffindor vs Rest
   - Hufflepuff vs Rest
   - Ravenclaw vs Rest
   - Slytherin vs Rest
   
   For prediction, we run all 4 classifiers and choose the house with the highest probability.

4. **Numerical Stability**: 
   - Sigmoid input is clipped to [-500, 500] to prevent overflow
   - Probabilities are clipped away from 0 and 1 to prevent log(0)

## Feature Selection

Based on the pair plot analysis, all 12 course features are used:
- Astronomy
- Herbology
- Divination
- Muggle Studies
- Ancient Runes
- History of Magic
- Transfiguration
- Potions
- Care of Magical Creatures
- Charms
- Flying
- Defense Against the Dark Arts

## Performance

The model achieves the required accuracy of ≥98% on the test set (evaluated during peer-evaluation).

## Algorithm Implemented

- Batch Gradient Descent

## Author

jainavas

École 42 - Madrid

## Acknowledgments

Inspired by the famous Sorting Hat from Harry Potter, recreated with the power of Machine Learning.