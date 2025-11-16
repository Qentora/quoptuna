# QuOptuna Optimizer Architecture Documentation

## Current State: Simulated vs Real Implementation

### ⚠️ **Important: The Optimizer Page is Currently Simulated**

Yes, you're absolutely right! The optimizer and SHAP analysis are currently **simulated** - they're not actually running real computations. This is why the process feels too fast.

---

## What's Real vs Simulated

### ✅ **Real Backend Services (Exist and Work)**

Located in `/backend/app/services/workflow_service.py`:

1. **Real Optuna Optimization** (lines 270-311)
   - Uses `quoptuna.Optimizer` class
   - Runs actual hyperparameter optimization
   - Saves results to SQLite database
   - Returns best trial, parameters, and study info

2. **Real SHAP Analysis** (lines 313-344)
   - Uses `quoptuna.XAI` class
   - Generates actual SHAP plots (bar, beeswarm, violin, waterfall)
   - Calculates real feature importance
   - Creates visual plots

3. **UCI Dataset Fetching** (lines 131-147)
   - ✅ **THIS IS REAL NOW** - Uses `ucimlrepo` library
   - Fetches actual datasets from UCI repository
   - Returns real column names and data

4. **Data Preparation** (lines 195-219)
   - Real train/test split
   - Real data scaling
   - Real label encoding

### ❌ **Simulated Frontend Code**

Located in `/frontend/src/pages/Optimizer.tsx`:

1. **Simulated Optimization** (lines 259-301)
   ```typescript
   // This is FAKE - just a setTimeout loop
   for (let i = 1; i <= totalTrials; i++) {
     await new Promise((resolve) => setTimeout(resolve, 50));
     setCurrentTrial(i);
     setProgress((i / totalTrials) * 100);
   }

   // Mock results - not real
   const mockResults = {
     bestValue: 0.9234,  // Hardcoded!
     bestParams: { ... } // Hardcoded!
   };
   ```

2. **Simulated SHAP Analysis** (lines 455-477)
   ```typescript
   // This is FAKE - just random numbers
   const mockSHAPData = {
     featureImportance: features.map(feature => ({
       feature,
       importance: Math.random() * 0.5 + 0.1  // Random!
     }))
   };
   ```

3. **Simulated Report Generation** (lines 623-686)
   ```typescript
   // This is a template string - not AI generated
   const report = `# Optimization Analysis Report...`;
   ```

### 🚧 **API Endpoints are Stubs**

Located in `/backend/app/api/v1/`:

- `/api/v1/optimize` - All TODOs, not implemented
- `/api/v1/analysis/shap` - All TODOs, not implemented
- `/api/v1/analysis/report` - All TODOs, not implemented

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│  /frontend/src/pages/Optimizer.tsx                              │
│                                                                   │
│  Step 1: Dataset Selection                                       │
│    ├─ Upload CSV         ────┐                                  │
│    └─ Select UCI Dataset ────┼─── ✅ REAL: Calls backend API   │
│                               │                                  │
│  Step 2: Features Selection   │                                  │
│    └─ Select columns      ────┤ ❌ STORED IN UI STATE ONLY     │
│                               │                                  │
│  Step 3: Configuration        │                                  │
│    └─ Study name, trials  ────┤ ❌ STORED IN UI STATE ONLY     │
│                               │                                  │
│  Step 4: Optimization         │                                  │
│    └─ Start Optimization  ────┼─── ❌ SIMULATED - setTimeout   │
│                               │                                  │
│  Step 5: SHAP Analysis        │                                  │
│    └─ Generate SHAP       ────┼─── ❌ SIMULATED - Random data  │
│                               │                                  │
│  Step 6: Generate Report      │                                  │
│    └─ AI Report           ────┘─── ❌ SIMULATED - Template     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓ HTTP
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND API LAYER                            │
│  /backend/app/api/v1/                                            │
│                                                                   │
│  ✅ /data/uci/{id}        → fetch_uci_dataset()                 │
│  ✅ /data/upload          → upload_dataset()                    │
│  🚧 /optimize             → start_optimization() [TODO]         │
│  🚧 /optimize/{id}        → get_optimization_status() [TODO]    │
│  🚧 /analysis/shap        → generate_shap_analysis() [TODO]     │
│  🚧 /analysis/report      → generate_ai_report() [TODO]         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND SERVICES LAYER                        │
│  /backend/app/services/workflow_service.py                      │
│                                                                   │
│  ✅ REAL IMPLEMENTATION EXISTS:                                 │
│                                                                   │
│  WorkflowExecutor                                                │
│    ├─ _execute_data_uci()           [REAL: Uses ucimlrepo]     │
│    ├─ _execute_optimization()       [REAL: Uses quoptuna.Optimizer]
│    ├─ _execute_shap_analysis()      [REAL: Uses quoptuna.XAI]  │
│    ├─ _execute_train_test_split()   [REAL: sklearn]            │
│    └─ _execute_generate_report()    [TODO: Needs LLM]          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CORE QUOPTUNA LIBRARY                         │
│  (Installed as dependency)                                       │
│                                                                   │
│  ✅ quoptuna.Optimizer                                           │
│     └─ optimize() → Runs Optuna study with quantum models       │
│                                                                   │
│  ✅ quoptuna.XAI                                                 │
│     └─ SHAP analysis, plots (bar, beeswarm, violin, waterfall)  │
│                                                                   │
│  ✅ quoptuna.DataPreparation                                     │
│     └─ Train/test split, scaling, encoding                      │
│                                                                   │
│  ✅ quoptuna.create_model()                                      │
│     └─ Quantum and classical models                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## How UI Triggers Backend Services

### Current Flow (What Actually Happens)

#### ✅ Dataset Selection (Working)
```
User clicks "Iris" in modal
    ↓
frontend/src/pages/Optimizer.tsx:handleUCISelect(53)
    ↓
fetchUCIDataset(53) in frontend/src/lib/api.ts
    ↓
HTTP POST to /api/v1/data/uci/53
    ↓
backend/app/api/v1/data.py:fetch_uci_dataset(53)
    ↓
Uses ucimlrepo.fetch_ucirepo(id=53)
    ↓
Returns: { dataset_id: "53", name: "Iris", rows: 150, columns: [...] }
    ↓
Stored in workflowData.dataset state
```

#### ❌ Optimization (Currently Simulated)
```
User clicks "Start Optimization"
    ↓
frontend/src/pages/Optimizer.tsx:startOptimization()
    ↓
[CURRENTLY: setTimeout loop creating fake progress]
    ↓
[SHOULD BE: HTTP POST to /api/v1/optimize]
    ↓
[SHOULD BE: Backend runs real Optuna optimization]
```

#### ❌ SHAP Analysis (Currently Simulated)
```
User clicks "Generate SHAP Analysis"
    ↓
frontend/src/pages/Optimizer.tsx:generateSHAP()
    ↓
[CURRENTLY: Random numbers for feature importance]
    ↓
[SHOULD BE: HTTP POST to /api/v1/analysis/shap]
    ↓
[SHOULD BE: Backend generates real SHAP plots]
```

---

## What Services Backend Uses

### 1. **Optuna** (Hyperparameter Optimization)
- Library: `optuna>=4.0.0`
- Purpose: Bayesian optimization framework
- Used in: `workflow_service.py:_execute_optimization()`
- Features:
  - Pruning (early stopping of bad trials)
  - Visualization of optimization history
  - SQLite database storage
  - Multiple samplers (TPE, CMA-ES, etc.)

### 2. **SHAP** (Explainable AI)
- Library: `shap>=0.46.0`
- Purpose: Explain model predictions
- Used in: `workflow_service.py:_execute_shap_analysis()`
- Features:
  - Feature importance calculation
  - Multiple plot types (bar, beeswarm, violin, waterfall)
  - Works with any ML model

### 3. **PennyLane** (Quantum Machine Learning)
- Library: `pennylane>=0.39.0`
- Purpose: Quantum computing and quantum ML
- Used in: `quoptuna.create_model()`
- Features:
  - Variational Quantum Circuits (VQC)
  - Data Reuploading
  - Quantum kernels
  - Hybrid quantum-classical models

### 4. **Scikit-learn** (Classical ML)
- Library: `scikit-learn>=1.5.0`
- Purpose: Classical machine learning
- Used in: `quoptuna.DataPreparation`, models
- Features:
  - Train/test split
  - Data scaling (StandardScaler)
  - Label encoding
  - Classical models (SVM, RandomForest, etc.)

### 5. **UCI ML Repository**
- Library: `ucimlrepo>=0.0.3`
- Purpose: Access to 600+ datasets
- Used in: `data.py:fetch_uci_dataset()`, `workflow_service.py`
- Features:
  - Fetch datasets by ID
  - Automatic feature/target separation
  - Metadata included

### 6. **Pandas & NumPy**
- Libraries: `pandas>=2.2.0`, `numpy>=1.24.0`
- Purpose: Data manipulation
- Used in: All data processing steps
- Features:
  - DataFrame operations
  - Data cleaning
  - Statistical analysis

---

## Implementation Status Summary

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Dataset Selection** | ✅ Complete | `frontend/Optimizer.tsx`, `backend/data.py` | Fully working |
| **Features Selection** | ✅ UI Only | `frontend/Optimizer.tsx` | Works but not persisted to backend |
| **Configuration** | ✅ UI Only | `frontend/Optimizer.tsx` | Works but not persisted to backend |
| **Optimization** | ⚠️ Backend Only | `backend/workflow_service.py` | Real code exists but UI uses mock |
| **SHAP Analysis** | ⚠️ Backend Only | `backend/workflow_service.py` | Real code exists but UI uses mock |
| **Report Generation** | ❌ Partial | `frontend/Optimizer.tsx` | UI has template, backend needs LLM |
| **Optimize API** | 🚧 TODO | `backend/optimize.py` | Endpoints exist but not implemented |
| **Analysis API** | 🚧 TODO | `backend/analysis.py` | Endpoints exist but not implemented |

---

## Why It's Fast (Simulated)

The optimizer completes in ~5 seconds because:

1. **Optimization**: Just a `setTimeout(50ms)` per "trial"
   - Real Optuna: 1-10 minutes for 100 trials
   - Simulated: 5 seconds (50ms × 100 trials)

2. **SHAP Analysis**: Just `Math.random()`
   - Real SHAP: 30 seconds to 5 minutes
   - Simulated: Instant

3. **No Model Training**: No actual ML models are trained
   - Real training: Minutes to hours
   - Simulated: 0 seconds

---

## Next Steps to Connect UI to Real Backend

See `IMPLEMENTATION_ROADMAP.md` for detailed steps to:
1. Implement `/api/v1/optimize` endpoints
2. Implement `/api/v1/analysis` endpoints
3. Update frontend to call real APIs
4. Add progress tracking with WebSockets
5. Implement AI report generation with LLM

---

**Generated:** 2025-11-16
**Status:** Current implementation uses simulated data in UI, but real services exist in backend
