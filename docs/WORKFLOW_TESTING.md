# Workflow Testing Guide

## Overview

This document describes the current state of the optimizer workflow, known issues, and how to test it properly.

## Recent Fixes

### Data Format Handling (2025-11-16)

Fixed critical data format mismatches between optimization and SHAP analysis:

1. **Optimizer** expects numpy arrays for `train_x`, `train_y`, `test_x`, `test_y`
2. **XAI** (SHAP) expects pandas DataFrames for the same data
3. **Model.fit()** requires numpy arrays

**Solution**: The workflow now:
- Stores DataFrames from DataPreparation
- Converts to numpy arrays when passing to Optimizer
- Retains DataFrames in optimization results
- Converts to numpy arrays only for model.fit() in SHAP analysis
- Passes DataFrames to XAI constructor

## Known Issues

### 1. PennyLane Quantum Device Deprecation

**Error**: `pennylane.exceptions.DeviceError: Device default.qubit.jax does not exist`

**Cause**: Quantum model classes (DataReuploadingClassifier, DressedQuantumCircuitClassifier, etc.) are configured to use `default.qubit.jax`, which is deprecated in newer PennyLane versions.

**Impact**:
- Quantum models fail during optimization
- Classical models (SVC, MLPClassifier, Perceptron) still work
- Optuna marks failed trials with value 0.0 and continues
- Best trial is selected from successful classical models

**Workaround**:
- Use small number of trials (10-20) to reduce quantum model failures
- Or modify quantum model classes to use `default.qubit` instead
- The workflow will still complete successfully with classical models

### 2. sklearn Parameter Validation

**Error**: `The 'learning_rate' parameter of MLPClassifier must be a str among {'constant', 'adaptive', 'invscaling'}`

**Cause**: Optuna search space suggests numeric learning rates, but MLPClassifier expects string values

**Impact**: Some MLPClassifier trials fail with 0.0 value

**Workaround**: Ignored - Optuna will try other parameters

## Testing the Workflow

### Recommended Test Process

1. **Select UCI Dataset**:
   - Choose a simple dataset (Wine, Iris recommended)
   - Use small number of features (< 15)
   - Binary classification preferred

2. **Configure Optimization**:
   - Study name: `test_wine` or similar
   - Database name: `test_wine` or similar
   - **Num trials: 10-20** (not 100) to reduce quantum errors

3. **Run Optimization**:
   - Expect some trial failures (quantum models)
   - Watch for successful classical models
   - Should complete in 1-2 minutes for 10 trials

4. **Generate SHAP Analysis**:
   - Should work with the best classical model
   - Generates bar, beeswarm, and waterfall plots
   - Returns feature importance rankings

### Example Working Configuration

```json
{
  "dataset": {
    "id": "109",
    "name": "Wine",
    "source": "uci"
  },
  "features": {
    "selectedFeatures": ["Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash"],
    "targetColumn": "class"
  },
  "configuration": {
    "studyName": "wine_test",
    "databaseName": "wine_test",
    "numTrials": 10
  }
}
```

## Reference Notebooks

The workflow implementation is based on these working examples:

1. **`experiments/basic_dataset_test/test_shap.ipynb`**
   - Shows complete workflow: load study → create model → fit → XAI → plots
   - Demonstrates correct data format handling
   - Example with Statlog dataset

2. **`experiments/basic_dataset_test/test_new_data_test.ipynb`**
   - Shows UCI dataset fetching and preparation
   - Demonstrates DataPreparation usage
   - Shows conversion to numpy arrays for Optimizer

## Workflow Steps (Technical)

### 1. Dataset Selection
```
Frontend → POST /api/v1/data/uci/{dataset_id}
Backend: fetch_ucirepo(id) → return columns
```

### 2. Feature Selection
```
Frontend: User selects features + target
State stored in workflowData.features
```

### 3. Optimization
```
Frontend → POST /api/v1/optimize
Backend:
  1. Create workflow nodes (data, features, split, model, optuna, optimize)
  2. Execute workflow in topological order:
     - data-uci: Fetch dataset
     - feature-selection: Select specified features
     - train-test-split: Use DataPreparation → returns DataFrames
     - optimize: Convert to numpy → create Optimizer → run trials
  3. Return optimization results with DataFrames intact
```

### 4. SHAP Analysis
```
Frontend → POST /api/v1/analysis/shap
Backend:
  1. Load best trial from Optuna study
  2. Recreate model with best parameters
  3. Convert DataFrames to numpy → fit model
  4. Create XAI with model + DataFrames
  5. Generate SHAP plots
  6. Calculate feature importance
  7. Return plots + importance
```

## Architecture

```
┌─────────────────┐
│   Frontend UI   │
│  (React/TS)     │
└────────┬────────┘
         │
         ↓ API Calls
┌─────────────────┐
│  FastAPI        │
│  /api/v1/       │
│  - optimize     │
│  - analysis     │
│  - data         │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ WorkflowService │
│  Executes nodes │
│  in order       │
└────────┬────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    ↓         ↓         ↓          ↓
┌─────────┐ ┌─────┐ ┌──────┐ ┌────────┐
│ Optuna  │ │ XAI │ │ Data │ │ Models │
│ Optimizer│ │SHAP │ │ Prep │ │        │
└─────────┘ └─────┘ └──────┘ └────────┘
```

## Success Criteria

✅ **Optimization completes** with best_value > 0
✅ **At least one trial succeeds** (classical model)
✅ **SHAP analysis generates** without 500 error
✅ **Feature importance** is non-empty array
✅ **Plots** are base64-encoded images

## Troubleshooting

### "All trials returning 0.0"
- Check if data has NaN values
- Verify target column has correct labels
- Try different dataset

### "SHAP 500 error"
- Check backend logs for specific error
- Verify optimization completed successfully
- Confirm database file exists in `backend/db/`

### "No plots generated"
- Check if model has predict_proba method
- Verify XAI config subset_size is reasonable
- Check backend logs for plot generation errors

## Next Steps

To fully resolve quantum model issues, would need to:

1. Update quantum model classes to use `default.qubit`
2. Or install `pennylane-qiskit` for Jax support
3. Or restrict Optuna search space to only classical models

For now, the workflow is **functional** with classical models and can demonstrate the complete optimization + SHAP analysis pipeline.
