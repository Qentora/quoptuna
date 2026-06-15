# QuOptuna Frontend - Quick Reference Guide

## Application Flow at a Glance

```
START (http://localhost:8501)
  ↓
┌────────────────────────────────────────────────────────────┐
│                    HOME PAGE (main_page.py)                │
│                     Welcome & Overview                      │
│  - Project description                                     │
│  - 3-column feature highlights                             │
│  - Step-by-step workflow guide                             │
│  - Tips & best practices                                   │
└────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────┐
│        PAGE 1: DATASET SELECTION (1_dataset_selection.py)  │
│                 Upload & Prepare Data                      │
│                                                             │
│  TAB 1: UCI ML Repository                                 │
│    - Select popular dataset or custom ID                  │
│    - Metadata display (instances, features, area)         │
│                                                             │
│  TAB 2: Custom CSV Upload                                 │
│    - File uploader widget                                 │
│                                                             │
│  Configuration (appears after load):                      │
│    ├─ Data Preview: Head (10 rows) + statistics          │
│    ├─ Target Selection: Dropdown from columns             │
│    ├─ Feature Selection: Multiselect (auto-exclude target)│
│    ├─ Target Transformation:                              │
│    │   ├─ Map unique_value[0] → -1                       │
│    │   └─ Map unique_value[1] → 1                        │
│    ├─ Missing value handling: Drop NaN rows              │
│    └─ Save Configuration → data/{dataset_name}.csv       │
│                                                             │
│  Session State:                                            │
│    dataset_df, file_path, target_column, feature_columns  │
└────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────┐
│      PAGE 2: OPTIMIZATION (2_optimization.py)              │
│        Data Preparation & Hyperparameter Tuning            │
│                                                             │
│  SECTION 1: Data Preparation                              │
│    ├─ Load CSV from file_path (Page 1)                   │
│    ├─ DataPreparation(file_path, x_cols, y_col)         │
│    │   ├─ Read CSV                                       │
│    │   ├─ StandardScaler.fit_transform(X_train)          │
│    │   ├─ Train/test split (75/25, random_state=42)      │
│    │   └─ Map y to {-1, 1}                               │
│    ├─ Return: data_dict = {                              │
│    │   train_x: array, test_x: array,                    │
│    │   train_y: array, test_y: array                     │
│    │ }                                                     │
│    └─ Display: Train samples, Test samples               │
│                                                             │
│  SECTION 2: Optimization Setup                            │
│    ├─ Database Name: text_input (default: dataset_name)  │
│    ├─ Study Name: text_input (default: dataset_name)     │
│    ├─ Number of Trials: slider (10-200, default: 100)    │
│    └─ Run Button: Start Optimization                      │
│                                                             │
│  SECTION 3: Optimizer Execution                           │
│    ├─ Optimizer(db_name, study_name, data=data_dict)    │
│    ├─ optimize(n_trials=n_trials)                        │
│    │   ├─ Loop n_trials times:                           │
│    │   │   ├─ Suggest hyperparameters (TPE sampler)      │
│    │   │   ├─ Create model from 15 types                 │
│    │   │   ├─ model.fit(train_x, train_y)               │
│    │   │   ├─ Evaluate: F1, accuracy                     │
│    │   │   ├─ Store metrics in trial.user_attrs          │
│    │   │   └─ Save to SQLite db/{db_name}.db             │
│    │   └─ Return: (study, best_trials)                   │
│    └─ Display: Best 5 trials with metrics                │
│                                                             │
│  Session State:                                            │
│    data_dict, optimizer, study, best_trials, db_name      │
└────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────┐
│         PAGE 3: SHAP ANALYSIS (3_shap_analysis.py)         │
│      Model Explanation & AI-Powered Reporting              │
│                                                             │
│  STEP 1: Trial Selection                                  │
│    ├─ Dropdown: Select from best_trials                  │
│    ├─ Display: F1 scores, model type, all parameters     │
│    └─ Session State: selected_trial                      │
│                                                             │
│  STEP 2: Model Training                                   │
│    ├─ Extract: params = selected_trial.params             │
│    ├─ Create: model = create_model(**params)             │
│    ├─ Train: model.fit(train_x, train_y)                 │
│    └─ Session State: trained_model                       │
│                                                             │
│  STEP 3: SHAP Configuration & Analysis                    │
│    ├─ Checkbox: Use Probability (default: True)          │
│    ├─ Checkbox: Use Subset (default: True)               │
│    ├─ Slider: Subset Size (10-200, default: 50)          │
│    ├─ Create: XAI(model, data, XAIConfig(...))           │
│    │   ├─ Initialize SHAP Explainer (lazy-loaded)        │
│    │   ├─ Calculate SHAP values                          │
│    │   └─ Compute performance metrics                    │
│    └─ Session State: xai                                 │
│                                                             │
│  STEP 4: Visualizations (6 Tabs)                          │
│    ├─ TAB 1: Bar Plot                                    │
│    │   └─ xai.get_plot("bar", max_display=10)           │
│    ├─ TAB 2: Beeswarm Plot                               │
│    │   └─ xai.get_plot("beeswarm", max_display=10)      │
│    ├─ TAB 3: Violin Plot                                 │
│    │   └─ xai.get_plot("violin", max_display=10)        │
│    ├─ TAB 4: Heatmap                                     │
│    │   └─ xai.get_plot("heatmap", max_display=50)       │
│    ├─ TAB 5: Waterfall Plot                              │
│    │   ├─ Slider: Select sample index                   │
│    │   └─ xai.get_plot("waterfall", index=idx)          │
│    └─ TAB 6: Confusion Matrix                            │
│        └─ xai.plot_confusion_matrix() → matplotlib fig   │
│                                                             │
│  STEP 5: AI-Powered Report Generation                     │
│    ├─ LLM Provider: Selectbox (google/openai/anthropic)  │
│    ├─ API Key: password_input                            │
│    ├─ Model Name: text_input                             │
│    ├─ Dataset Context (optional):                        │
│    │   ├─ URL, Description, Features, Target            │
│    │   └─ Stored in dataset_info dict                    │
│    ├─ Generate Report:                                   │
│    │   ├─ xai.get_report() → dict of metrics             │
│    │   ├─ Generate SHAP plot images                      │
│    │   ├─ xai.generate_report_with_langchain(...)        │
│    │   │   ├─ Initialize ChatGoogleGenerativeAI/OpenAI   │
│    │   │   ├─ Create multimodal prompt                   │
│    │   │   ├─ Include base64-encoded plot images         │
│    │   │   └─ Return: markdown report string             │
│    │   └─ Display report + Download button               │
│    └─ Session State: report, shap_images                 │
└────────────────────────────────────────────────────────────┘
```

---

## Page-by-Page Functionality Matrix

| Page | File | Primary Class | Inputs | Outputs | Key Operations |
|------|------|---------------|--------|---------|-----------------|
| 1 | `1_dataset_selection.py` | N/A (utilities) | CSV or UCI ID | `file_path`, `dataset_df` | Load, preview, transform, save |
| 2 | `2_optimization.py` | `DataPreparation`, `Optimizer` | `file_path`, trial count | `best_trials`, `db_name` | Split, normalize, optimize |
| 3 | `3_shap_analysis.py` | `XAI` | `best_trials`, `data_dict` | `report`, `shap_images` | Explain, visualize, generate |

---

## Model Support Matrix

### 26 Total Models

| Category | Count | Examples |
|----------|-------|----------|
| Quantum Models | 18 | DataReuploading, CircuitCentric, QuantumKitchenSinks, DressedQuantumCircuit, QuantumMetricLearner, QuantumBoltzmannMachine, TreeTensorClassifier, QuanvolutionalNeuralNetwork, WeiNet, Separable variants, Convolutional variants, IQP, ProjectedQuantumKernel |
| Classical Models | 8 | SVC, LinearSVC, MLPClassifier, Perceptron |

### Key Parameters by Model Type

**All Quantum Models:**
- `max_vmap`: [1]
- `batch_size`: [32]
- `learning_rate`: [0.001, 0.01, 0.1]
- `n_layers`: [1, 5, 10]
- Model-specific: observable_type, n_qfeatures, repeats, C, gamma, etc.

**Classical Models:**
- `C` (SVC): [0.1, 1, 10, 100]
- `gamma` (SVC): [0.001, 0.01, 0.1, 1]
- `alpha` (MLP): [0.01, 0.001, 0.0001]
- `hidden_layer_sizes` (MLP): [(100,), (10,10,10,10), (50,10,5)]

---

## External Service Integration Points

| Service | Library | Purpose | Page Used | API Required |
|---------|---------|---------|-----------|--------------|
| UCI ML Repository | `ucimlrepo` | Dataset download | 1 | No |
| Optuna | `optuna` | Hyperparameter tuning | 2 | No |
| SQLite | `sqlite3` | Trial persistence | 2 | No |
| Scikit-learn | `sklearn` | Classical ML & metrics | 2, 3 | No |
| PennyLane | `pennylane` | Quantum circuits | 2 | No |
| SHAP | `shap` | Explainability | 3 | No |
| LangChain | `langchain` | LLM integration | 3 | Yes |
| Google GenAI | `langchain_google_genai` | Gemini models | 3 | Yes |
| OpenAI | `langchain_openai` | GPT models | 3 | Yes |

---

## Session State Initialization & Flow

### Initialization (app.py)
```python
initialize_session_state()
session_defaults = {
    "uploaded_file": None,
    "file_location": None,
    "x_columns": None,
    "y_column": None,
    "DB_NAME": None,
    "study_name": None,
    "n_trials": 100,
    "optimizer": None,
    "process_running": False,
    "start_visualization": False,
}
```

### Per-Page Updates
- **Page 1:** Adds dataset_loaded, dataset_df, file_path, target_column, feature_columns
- **Page 2:** Adds data_dict, study, best_trials, optimization_complete
- **Page 3:** Adds selected_trial, trained_model, xai, report, shap_images

---

## Data Transformation Pipeline

```
INPUT: CSV File
  ↓
STEP 1: Read CSV
  ↓
STEP 2: Select Columns (X_cols, y_col)
  ↓
STEP 3: Handle Missing Values (dropna)
  ↓
STEP 4: Transform Target
  unique_class[0] → -1
  unique_class[1] → 1
  ↓
STEP 5: Normalize Features
  StandardScaler.fit_transform(X)
  ↓
STEP 6: Train/Test Split
  X_train (75%), X_test (25%)
  y_train (75%), y_test (25%)
  ↓
OUTPUT: data_dict = {
    "train_x": numpy array,
    "test_x": numpy array,
    "train_y": numpy array with {-1, 1},
    "test_y": numpy array with {-1, 1}
}
```

---

## Visualization Types Reference

### SHAP Plots (Page 3)

| Plot Type | Method | Purpose | Parameters |
|-----------|--------|---------|------------|
| Bar | `xai.get_plot("bar")` | Global feature importance | max_display=10 |
| Beeswarm | `xai.get_plot("beeswarm")` | Feature value impact | max_display=10 |
| Violin | `xai.get_plot("violin")` | SHAP value distribution | max_display=10 |
| Heatmap | `xai.get_plot("heatmap")` | Instance-level analysis | max_display=50 |
| Waterfall | `xai.get_plot("waterfall")` | Single prediction breakdown | index=0 |
| Confusion Matrix | `xai.plot_confusion_matrix()` | Classification performance | standard |

### Optuna Plots (Visualization in support.py)

| Plot Type | Method | Purpose |
|-----------|--------|---------|
| Timeline | `optuna.visualization.plot_timeline()` | Trial execution timeline |
| Parameter Importance | `optuna.visualization.plot_param_importances()` | Which params matter most |
| Optimization History | `optuna.visualization.plot_optimization_history()` | F1 score over trials |

### Performance Metrics Display

| Metric | Method | Formula |
|--------|--------|---------|
| F1 Score | `f1_score(y_true, y_pred)` | 2 * (precision * recall) / (precision + recall) |
| Accuracy | `accuracy_score(y_true, y_pred)` | Correct / Total |
| Precision | `precision_score(y_true, y_pred)` | TP / (TP + FP) |
| Recall | `recall_score(y_true, y_pred)` | TP / (TP + FN) |
| MCC | `matthews_corrcoef(y_true, y_pred)` | Correlation coefficient |
| AUC | `roc_auc_score(y_true, y_proba)` | Area under ROC curve |

---

## Error Handling & Validation

### Key Validation Points

1. **Page 1 (Dataset Selection)**
   - CSV format validation
   - Binary target verification
   - Missing value detection
   - Feature column selection (minimum 1 required)

2. **Page 2 (Optimization)**
   - File path verification
   - Data dict completeness check
   - Model instantiation safety (try/except)
   - Training error recovery

3. **Page 3 (SHAP Analysis)**
   - Trial selection requirement
   - Model training exception handling
   - SHAP computation fallback
   - API key validation for report generation

### Exception Handling Pattern
```python
try:
    # Operation
except Exception as e:
    st.error(f"Error message: {e}")
    st.exception(e)  # Debug info
    return False
```

---

## File I/O Operations

| Operation | Location | Path | Purpose |
|-----------|----------|------|---------|
| Upload CSV | support.py | `./uploaded_data/{filename}` | Temporary data storage |
| Prepared Dataset | 1_dataset_selection.py | `data/{dataset_name}.csv` | Processed data input for Opt |
| Optuna Database | optimizer.py | `db/{db_name}.db` | SQLite trial persistence |
| Plot Output | 3_shap_analysis.py | `outputs/{plot_name}.png` | User-saved visualizations |
| Report Download | 3_shap_analysis.py | Browser download | Markdown file |

---

## Key Metrics Tracked During Optimization

For **each trial**, stored in `trial.user_attrs`:

```python
{
    "Quantum_f1_score": float,      # F1 for quantum models
    "Quantum_accuracy": float,      # Accuracy for quantum models
    "Quantum_score": float,         # Generic score for quantum
    "Classical_f1_score": float,    # F1 for classical models
    "Classical_accuracy": float,    # Accuracy for classical models
    "Classical_score": float,       # Generic score for classical
}
```

Plus in trial object:
- `number`: Trial ID
- `state`: TrialState (COMPLETE, FAIL, RUNNING, etc.)
- `value`: Objective value (F1 score)
- `params`: All hyperparameters
- `datetime_start`, `datetime_complete`: Execution timing

---

## Configuration Files

### `.streamlit/config.toml`
```toml
[theme]
base = "dark"
primaryColor = "#8e44ad"              # Purple accent
backgroundColor = "#121212"           # Dark background
secondaryBackgroundColor = "#1c1c1c"  # Panel background
textColor = "#dcdcdc"                 # Light text
font = "sans serif"
```

### Custom CSS (app.py)
- `.main-title`: Purple, 3em, center-aligned
- `.description`: Light grey, 1.2em, center-aligned
- `.stButton>button`: Purple background, hover effect

---

## Typical Workflow Timing

| Stage | Operation | Typical Duration |
|-------|-----------|------------------|
| 1 | Load UCI dataset | 1-5 seconds |
| 1 | Configure columns | Instant |
| 2 | Prepare data | <1 second |
| 2 | Optimize (100 trials) | 5-60 minutes (model-dependent) |
| 3 | Train model | 1-10 seconds |
| 3 | Calculate SHAP values | 10-60 seconds |
| 3 | Generate report (with LLM) | 30-120 seconds |

---

## Common Use Cases

### Use Case 1: Quick Exploration
1. Load popular UCI dataset
2. Run 20-50 trials
3. View best trial's SHAP plots

### Use Case 2: Comprehensive Analysis
1. Upload custom CSV
2. Run 100-200 trials
3. Generate AI report with all SHAP visualizations
4. Download markdown report

### Use Case 3: Quantum vs Classical Comparison
1. Load dataset
2. Run optimization (tracks both model types)
3. Compare F1 scores in results table
4. Train and analyze best quantum model
5. Generate report with quantum-specific insights

---

## Tips for Users

### Performance Optimization
- Use smaller subset for SHAP (50 samples) instead of full dataset
- Start with 50 trials, then run more if needed
- Use probability predictions for faster SHAP calculation

### Better Results
- Ensure sufficient data (100+ samples minimum)
- Balance dataset classes when possible
- Run 100+ trials for reliable optimization

### Report Generation
- Use faster models (Gemini Flash) for quick feedback
- Use advanced models (GPT-4, Claude 3) for detailed analysis
- Provide dataset description for context

