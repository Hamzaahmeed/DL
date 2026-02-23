# Neural Networks for Binary Classification: Diabetes Diagnosis

**Author:** Hamza Ahmed  
**ID:** 1210219  

---

## 📋 Project Overview

This project implements a **binary classification neural network** to predict diabetes diagnosis using the **Pima Indians Diabetes Dataset**. The goal is to demonstrate fundamental deep learning concepts including data preprocessing, model building, training optimization, and performance evaluation with medical context.

**Key Dataset:** 768 patient records with 8 clinical measurements  
**Target:** Diabetes diagnosis (0 = No Diabetes, 1 = Diabetes)  
**Challenge:** Realistic classification problem (~75% max accuracy expected)

---

## 📊 Dataset Description

### Source
- **Origin:** UCI Machine Learning Repository / Kaggle
- **File:** `diabetes.csv` (768 samples, 8 features, 1 target)

### Features (8 Clinical Measurements)
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (2-hour OGTT), mg/dL
3. **BloodPressure** - Diastolic blood pressure, mmHg
4. **SkinThickness** - Triceps skin fold thickness, mm
5. **Insulin** - 2-hour serum insulin, mu U/ml
6. **BMI** - Body Mass Index, kg/m²
7. **DiabetesPedigreeFunction** - Genetic diabetes risk score
8. **Age** - Age in years

### Target Variable
- **Outcome:** Binary (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution:** 
  - No Diabetes: 500 samples (65.1%)
  - Diabetes: 268 samples (34.9%)

### Data Quality
**Note:** Several features (BloodPressure, SkinThickness, Insulin, BMI) contained zero values representing missing measurements. These were **imputed with median values** from non-zero entries.

---

## 🏗️ Model Architecture

### Original Model (Baseline)
```
Input Layer:        8 features
    ↓
Dense Layer 1:      32 neurons, ReLU activation
    ↓
Dense Layer 2:      16 neurons, ReLU activation
    ↓
Output Layer:       1 neuron, Sigmoid activation (probability)
```

**Parameters:** 833 total trainable parameters

**Rationale:**
- ReLU activations introduce non-linearity to capture complex patterns
- Sigmoid output produces probability (0-1) suitable for binary classification
- Progressive dimensionality reduction (8 → 32 → 16 → 1) for hierarchical feature learning

### Model v2 (Same Baseline)
For this implementation, Model v2 uses the same architecture as the original for comparison purposes. Students are encouraged to modify this architecture by:
- Adding/removing layers
- Changing neuron counts
- Adding Dropout regularization (e.g., `Dropout(0.3)`)
- Using different activation functions (tanh, elu, etc.)

---

##  Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | Adam (lr=0.001) | Adaptive learning rates, efficient convergence |
| Loss Function | Binary Crossentropy | Standard for binary classification |
| Metrics | Binary Accuracy | Percentage of correct predictions |
| Epochs | 100 (max) | Sufficient for convergence |
| Batch Size | 32 | Balance between speed and gradient stability |
| Validation Split | 20% | Monitor overfitting during training |
| Early Stopping | patience=5 | Stop if validation loss doesn't improve for 5 epochs |
| Train-Test Split | 80/20 | Standard split with stratification |
| Feature Scaling | StandardScaler | Mean=0, Std=1 normalization |

---

##  Results & Performance Comparison

### Original Model Performance
| Metric | Value |
|--------|-------|
| **Test Accuracy** | 73.38% |
| **Test Loss** | 0.4930 |
| **Precision** | 0.6383 |
| **Recall (Sensitivity)** | 0.5556 |
| **F1-Score** | 0.5941 |
| **Specificity** | 0.8300 |
| **Training Epochs** | 30 |

### Confusion Matrix (Original Model)
```
                Predicted
              No Diabetes  Diabetes
Actual No Diabetes    83        17       (TN=83, FP=17)
       Diabetes       24        30       (FN=24, TP=30)
```

**Interpretation:**
- **True Negatives (TN):** 83 - Correctly identified patients without diabetes
- **False Positives (FP):** 17 - Incorrectly predicted diabetes (unnecessary treatment)
- **False Negatives (FN):** 24 - ⚠️ **Missed diabetes cases (critical error)**
- **True Positives (TP):** 30 - Correctly identified diabetes cases

### Model v2 Performance
| Metric | Value | vs Original |
|--------|-------|------------|
| **Test Accuracy** | 74.03% | +0.65% |
| **Precision** | 0.6522 | +0.0139 |
| **Recall** | 0.5556 | Same |
| **F1-Score** | 0.6000 | +0.0059 |

---

##  Clinical Interpretation

### Medical Context
In diabetes screening, **false negatives are the most critical error type**:
- **False Negative (FN):** Predicted no diabetes → Actually has diabetes
  - **Risk:** Missed diagnosis, delayed treatment, health complications
  - **Our model:** 24 missed cases out of 54 actual positives (44.4% miss rate)

- **False Positive (FP):** Predicted diabetes → Actually no diabetes
  - **Risk:** Unnecessary treatment, patient stress, unnecessary medical costs
  - **Our model:** 17 false alarms out of 100 actual negatives (17% false alarm rate)

### Key Metrics Explained
- **Recall (Sensitivity):** Of patients who ACTUALLY have diabetes, we catch 55.56%
  - **Clinical Goal:** Maximize recall to minimize missed diagnoses
  - **Trade-off:** Lower recall means more false negatives (worse outcome)

- **Precision:** Of patients we PREDICT have diabetes, 63.83% actually do
  - **Clinical Goal:** Good precision reduces unnecessary treatments
  - **Note:** We're missing ~44% of diabetic patients (low recall)

- **F1-Score:** Harmonic mean balancing precision and recall
  - Current: 0.5941 (indicates room for improvement)

---

## 📁 Project Structure

```
DL/
├── diabetes.csv                              # Dataset (768 samples)
├── Notebook.ipynb                            # Main notebook with full analysis
├── Neural_Networks_Binary_Classification.ipynb  # Comprehensive lab notebook
├── README.md                                 # This file
└── results/                                  # (Optional) Save plots here
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── model_comparison.png
    └── metrics_table.csv
```


##  Step-by-Step Workflow

### 1. Data Loading & Exploration
- Load `diabetes.csv` (768 samples, 8 features)
- Handle missing values (zeros → median imputation)
- Explore class distribution and feature ranges

### 2. Data Preprocessing
- **Train-Test Split:** 80% training (614), 20% testing (154)
- **Stratification:** Preserve class distribution in both sets
- **Feature Scaling:** StandardScaler (mean=0, std=1)

### 3. Model Building
- Build sequential neural network: 8 → 32 → 16 → 1
- Compile with Adam optimizer and binary crossentropy loss

### 4. Model Training
- Train for max 100 epochs with batch size 32
- Use 20% validation split to monitor overfitting
- Applied EarlyStopping (patience=5) to stop when validation loss plateaus

##  Training Curves Analysis

### Observations from Original Model
- **Training Accuracy:** Reaches ~82% by epoch 30
- **Validation Accuracy:** Peaks at ~77% then plateaus
- **Gap Analysis:** 5% gap indicates some overfitting (watch for larger gaps)
- **Loss Convergence:** Both training and validation loss decrease smoothly
- **Early Stopping:** Model stopped at epoch 30 (no improvement for 5 epochs)

### Interpretation
- Model learned well but shows slight overfitting (expected with 768 samples)
- Adding regularization (Dropout, L2, or larger learning rate) could help
- Validation curves plateauing suggests model reached its capacity

---
Insights : 

1. **Data Imbalance:** 65% No Diabetes vs 35% Diabetes - stratified split preserves this
2. **Feature Scaling Critical:** Different feature ranges (Glucose 0-199, Age 21-81) require normalization
3. **Overfitting Concern:** With only 768 samples, regularization is important
4. **Medical Trade-off:** High recall means catching diabetes but more false alarms
5. **Early Stopping:** Prevents overfitting by monitoring validation loss





