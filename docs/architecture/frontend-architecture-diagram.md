# QuOptuna Frontend - Architecture Diagrams

## 1. High-Level Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT APPLICATION                       │
│                     (src/quoptuna/frontend/)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      SIDEBAR                                │  │
│  │  (sidebar.py + support.py)                                 │  │
│  │                                                             │  │
│  │  ├─ Data Upload Component                                 │  │
│  │  ├─ Database Configuration                               │  │
│  │  ├─ Study Name Setup                                     │  │
│  │  ├─ Trial Count Slider                                  │  │
│  │  └─ Visualization Control                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      MAIN CONTENT                            │  │
│  │                                                              │  │
│  │  HOME PAGE (main_page.py)                                  │  │
│  │  ├─ Welcome & Overview                                    │  │
│  │  ├─ Feature Highlights (3 columns)                        │  │
│  │  └─ Getting Started Guide                                 │  │
│  │                                                             │  │
│  │  PAGE 1: Dataset Selection (pages/1_dataset_selection.py) │  │
│  │  ├─ UCI Repository Browser                               │  │
│  │  ├─ Custom CSV Upload                                    │  │
│  │  └─ Data Configuration Panel                             │  │
│  │                                                             │  │
│  │  PAGE 2: Optimization (pages/2_optimization.py)           │  │
│  │  ├─ Data Preparation Module                              │  │
│  │  ├─ Optimization Configuration                           │  │
│  │  ├─ Trial Progress Display                               │  │
│  │  └─ Results Table                                         │  │
│  │                                                             │  │
│  │  PAGE 3: SHAP Analysis (pages/3_shap_analysis.py)         │  │
│  │  ├─ Trial Selector                                       │  │
│  │  ├─ Model Training Controls                              │  │
│  │  ├─ SHAP Configuration                                   │  │
│  │  ├─ 6 Visualization Tabs                                 │  │
│  │  └─ AI Report Generator                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         ↓                                   ↓                    ↓
    ┌─────────────┐              ┌──────────────────┐    ┌─────────────┐
    │  Backend    │              │    External      │    │  Storage    │
    │  Libraries  │              │    Services      │    │  & Files    │
    └─────────────┘              └──────────────────┘    └─────────────┘
```

---

## 2. Data Flow Architecture

```
┌─────────────┐
│  USER INPUT │
└──────┬──────┘
       │
       ├─ Dataset Source
       │  ├─ UCI Repository (fetch_ucirepo)
       │  └─ Custom CSV Upload
       │
       ├─ Target Selection (Column)
       │
       ├─ Feature Selection (Multiselect)
       │
       └─ Transformation Parameters
           ├─ Missing value handling
           ├─ Target encoding (-1, 1)
           └─ File path

       ↓

┌──────────────────────────────────────────┐
│  PAGE 1: DATASET SELECTION               │
│  (1_dataset_selection.py)                │
│                                          │
│  fetch_uci_dataset() / upload_custom()  │
│  ├─ Load data (pandas DataFrame)       │
│  ├─ Preview & validate                 │
│  └─ Save to: data/{dataset_name}.csv   │
│                                          │
│  Session State: dataset_df, file_path   │
└──────────────┬───────────────────────────┘
               │
               │ (file_path)
               ↓

┌──────────────────────────────────────────────────────┐
│  PAGE 2: OPTIMIZATION                               │
│  (2_optimization.py)                                │
│                                                     │
│  DataPreparation(file_path, x_cols, y_col)         │
│  ├─ Read CSV                                       │
│  ├─ Feature scaling (StandardScaler)               │
│  ├─ Train/test split (75/25)                       │
│  └─ Output:                                        │
│      data_dict = {                                 │
│          'train_x': ndarray,                       │
│          'test_x': ndarray,                        │
│          'train_y': ndarray[-1,1],                │
│          'test_y': ndarray[-1,1]                  │
│      }                                             │
│                                                     │
│  Optimizer(db_name, study_name, data=data_dict)   │
│  ├─ optimize(n_trials)                            │
│  │   ├─ Loop n_trials:                            │
│  │   │   ├─ Suggest hyperparameters (TPE)         │
│  │   │   ├─ create_model(model_type, **params)    │
│  │   │   ├─ Train model                           │
│  │   │   ├─ Evaluate F1, accuracy                 │
│  │   │   ├─ Store metrics in trial.user_attrs     │
│  │   │   └─ Save to SQLite                        │
│  │   └─ Return (study, best_trials)               │
│  └─ Output to: db/{db_name}.db                    │
│                                                     │
│  Session State: data_dict, optimizer,              │
│                 best_trials, study                │
└──────────────┬──────────────────────────────────────┘
               │
               │ (best_trials, data_dict)
               ↓

┌─────────────────────────────────────────────────────┐
│  PAGE 3: SHAP ANALYSIS                             │
│  (3_shap_analysis.py)                              │
│                                                     │
│  Step 1: Trial Selection (dropdown)                │
│  └─ selected_trial = best_trials[idx]              │
│                                                     │
│  Step 2: Model Training                            │
│  ├─ params = selected_trial.params                 │
│  ├─ model = create_model(**params)                 │
│  ├─ model.fit(data_dict['train_x'],               │
│  │           data_dict['train_y'])                │
│  └─ Session State: trained_model                  │
│                                                     │
│  Step 3: SHAP Analysis                             │
│  ├─ XAI(model, data_dict, XAIConfig(...))         │
│  ├─ xai.get_shap_values()                         │
│  ├─ xai.get_explainer()                           │
│  └─ Session State: xai                            │
│                                                     │
│  Step 4: Visualizations                            │
│  ├─ xai.get_plot('bar') → base64 image            │
│  ├─ xai.get_plot('beeswarm') → base64 image       │
│  ├─ xai.get_plot('violin') → base64 image         │
│  ├─ xai.get_plot('heatmap') → base64 image        │
│  ├─ xai.get_plot('waterfall') → base64 image      │
│  └─ xai.plot_confusion_matrix() → matplotlib fig  │
│                                                     │
│  Step 5: AI Report Generation                      │
│  ├─ xai.get_report() → dict of metrics             │
│  ├─ Generate SHAP plot images                     │
│  ├─ xai.generate_report_with_langchain(           │
│  │       provider, api_key, model_name,            │
│  │       dataset_info)                             │
│  │   ├─ Initialize LLM client                     │
│  │   ├─ Create multimodal prompt                  │
│  │   ├─ Include base64-encoded images             │
│  │   └─ Return markdown report                    │
│  └─ Session State: report, shap_images            │
│                                                     │
│  OUTPUT:                                            │
│  ├─ SHAP visualizations (displayed)                │
│  └─ Markdown report (downloadable)                 │
└─────────────────────────────────────────────────────┘
```

---

## 3. Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND MODULES                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  app.py                                                    │
│  ├─ imports: main_page, sidebar, support                 │
│  └─ imports: st (streamlit)                              │
│                                                             │
│  main_page.py                                             │
│  ├─ imports: st                                          │
│  └─ pure display/documentation                            │
│                                                             │
│  sidebar.py                                               │
│  ├─ imports: st                                          │
│  ├─ imports: Optimizer                                   │
│  ├─ imports: preprocess_data                             │
│  ├─ imports: support module functions                    │
│  └─ orchestrates data upload & DB config                │
│                                                             │
│  support.py                                               │
│  ├─ imports: st, optuna, pandas                          │
│  ├─ exports: upload_and_display_data()                   │
│  ├─ exports: select_columns()                            │
│  ├─ exports: initialize_session_state()                  │
│  ├─ exports: update_plot()                               │
│  └─ shared visualization utilities                        │
│                                                             │
│  pages/1_dataset_selection.py                             │
│  ├─ imports: st, pandas, ucimlrepo                       │
│  ├─ imports: mock_csv_data (utility)                     │
│  ├─ functions:                                            │
│  │  ├─ fetch_uci_dataset()                              │
│  │  ├─ upload_custom_dataset()                          │
│  │  └─ configure_dataset()                              │
│  └─ standalone Streamlit page                            │
│                                                             │
│  pages/2_optimization.py                                  │
│  ├─ imports: st                                          │
│  ├─ imports: DataPreparation, Optimizer                  │
│  ├─ functions:                                            │
│  │  ├─ prepare_data()                                   │
│  │  ├─ run_optimization()                               │
│  │  └─ display_results()                                │
│  └─ standalone Streamlit page                            │
│                                                             │
│  pages/3_shap_analysis.py                                 │
│  ├─ imports: st, base64, io                              │
│  ├─ imports: XAI, XAIConfig                              │
│  ├─ imports: create_model                                │
│  ├─ functions:                                            │
│  │  ├─ select_trial()                                   │
│  │  ├─ train_model()                                    │
│  │  ├─ run_shap_analysis()                              │
│  │  ├─ display_shap_plots()                             │
│  │  ├─ generate_report()                                │
│  │  └─ save_plot()                                      │
│  └─ standalone Streamlit page                            │
│                                                             │
│  pages/shap.py (legacy)                                   │
│  ├─ imports: various backend modules                     │
│  └─ old SHAP workflow (not actively used)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓                                  ↓
    ┌──────────────────────┐      ┌──────────────────────┐
    │  BACKEND MODULES     │      │  EXTERNAL LIBRARIES  │
    ├──────────────────────┤      ├──────────────────────┤
    │                      │      │                      │
    │ Optimizer            │      │ streamlit (st)       │
    │ DataPreparation      │      │ pandas               │
    │ create_model()       │      │ numpy                │
    │ XAI                  │      │ optuna               │
    │ XAIConfig            │      │ sklearn              │
    │ preprocess_data()    │      │ shap                 │
    │ mock_csv_data()      │      │ pennylane            │
    │                      │      │ ucimlrepo            │
    │                      │      │ langchain            │
    │                      │      │ matplotlib           │
    │                      │      │ plotly               │
    └──────────────────────┘      └──────────────────────┘
```

---

## 4. State Management Flow

```
START: initialize_session_state() [app.py]
  │
  ├─ Set initial values for all keys
  └─ Create empty dict for session_state

     ↓

PAGE 1: Dataset Selection
  │
  ├─ dataset_loaded = False → True
  ├─ dataset_df = None → DataFrame
  ├─ dataset_name = None → "Australian_Credit_Approval"
  ├─ dataset_metadata = None → UCI metadata dict
  ├─ file_path = None → "data/Australian_Credit_Approval.csv"
  ├─ target_column = None → "target"
  └─ feature_columns = None → ["feat1", "feat2", ...]

     ↓

PAGE 2: Optimization
  │
  ├─ data_dict = None → {"train_x": ndarray, ...}
  ├─ db_name = None → "Australian_Credit_Approval"
  ├─ study_name = None → "Australian_Credit_Approval"
  ├─ optimizer = None → Optimizer instance
  ├─ study = None → Optuna Study object
  ├─ best_trials = None → [Trial1, Trial2, ...]
  └─ optimization_complete = False → True

     ↓

PAGE 3: SHAP Analysis
  │
  ├─ selected_trial = None → Trial object
  ├─ trained_model = None → Model instance
  ├─ xai = None → XAI instance
  ├─ report = None → "# Analysis Report\n..."
  └─ shap_images = None → {"bar": "data:image/png;base64,...", ...}

END: All state available across all pages
```

---

## 5. Backend Integration Points

```
                    FRONTEND (Streamlit)
                           ↓
            ┌──────────────────────────────────┐
            │    Python Library Imports         │
            └──────────────────────────────────┘
                  ↓           ↓          ↓
        ┌─────────────┐  ┌────────┐  ┌─────────────┐
        │  Optimizer  │  │ Data   │  │   XAI       │
        │             │  │Prepar  │  │             │
        │  optimize() │  │ation   │  │ get_plot()  │
        │   objective │  │        │  │ get_report()│
        │  log_attrs  │  │prepare │  │generate_rpt │
        └──────┬──────┘  └───┬────┘  └──────┬──────┘
               │             │              │
               ├─ Optuna ─────┼─ PennyLane  │
               │ (Study)      │ (Quantum)   │
               │              │             │
               └──SkLearn─────┴─ SHAP ──────┘
                  (Classical)    (Explainability)

  Storage:
  ├─ SQLite: db/{db_name}.db
  ├─ File: data/{dataset_name}.csv
  └─ Memory: st.session_state

  External APIs:
  ├─ UCI Repository (read-only)
  └─ LLM APIs (OpenAI, Google, Anthropic) [optional]
```

---

## 6. Request/Response Cycle for Key Operations

### Operation: Load UCI Dataset
```
User Action: Click "Load UCI Dataset"
  │
  ├─ fetch_ucirepo(id=143)
  │  └─ API call to UCI Repository
  │     └─ Returns: Dataset object
  │
  ├─ Extract X, y from dataset
  ├─ Combine: pd.concat([X, y], axis=1)
  │
  ├─ Store in st.session_state:
  │  ├─ dataset_df
  │  ├─ dataset_name
  │  └─ dataset_metadata
  │
  └─ Display:
     ├─ ✅ Success message
     ├─ Metadata expander (instances, features, area, tasks)
     └─ Data preview button
```

### Operation: Run Optimization
```
User Action: Click "Start Optimization"
  │
  ├─ DataPreparation.prepare_data()
  │  ├─ Read CSV from file_path
  │  ├─ StandardScaler.fit_transform()
  │  ├─ train_test_split(75/25)
  │  └─ Return data_dict
  │
  ├─ Optimizer(db_name, study_name, data=data_dict)
  │  └─ create_study(storage, sampler=TPE, study_name)
  │
  ├─ study.optimize(objective, n_trials=n_trials)
  │  │
  │  └─ For each trial:
  │     ├─ trial.suggest_categorical() [hyperparameters]
  │     ├─ create_model(**params)
  │     ├─ model.fit(train_x, train_y)
  │     ├─ f1 = f1_score(test_y, model.predict(test_x))
  │     ├─ trial.set_user_attr("Quantum_f1_score", f1)
  │     └─ return f1
  │
  ├─ study.best_trials → filtered top trials
  │
  ├─ Store in st.session_state:
  │  ├─ best_trials
  │  ├─ optimizer
  │  └─ optimization_complete = True
  │
  └─ Display:
     ├─ Progress bar (100%)
     └─ Best trials table with metrics
```

### Operation: Generate SHAP Report
```
User Action: Click "Generate Report"
  │
  ├─ Validate API key provided
  │
  ├─ XAI.get_report()
  │  └─ Compute all metrics:
  │     ├─ confusion_matrix()
  │     ├─ classification_report()
  │     ├─ roc_auc_score()
  │     ├─ f1_score()
  │     └─ etc...
  │
  ├─ Generate plot images:
  │  ├─ For plot_type in ["bar", "beeswarm", "violin", "heatmap"]:
  │  │  └─ xai.get_plot(plot_type) → base64 image
  │  ├─ For index in range(num_waterfall_plots):
  │  │  └─ xai.get_plot("waterfall", index=i) → base64 image
  │  └─ xai.plot_confusion_matrix() → base64 image
  │
  ├─ Initialize LLM client:
  │  └─ ChatGoogleGenerativeAI(api_key, model_name)
  │     OR ChatOpenAI(api_key, model_name)
  │
  ├─ Create multimodal prompt:
  │  ├─ System message: prompt.txt
  │  ├─ Human message: report dict
  │  └─ Human messages: [image for each plot]
  │
  ├─ LLM response:
  │  └─ response = chat(final_prompt)
  │     └─ Returns: markdown report string
  │
  ├─ Store in st.session_state:
  │  ├─ report
  │  └─ shap_images
  │
  └─ Display:
     ├─ Report markdown (rendered)
     ├─ Download button (report.md)
     └─ ✅ Success message
```

---

## 7. File Structure & I/O Diagram

```
PROJECT ROOT
│
├─ src/quoptuna/
│  ├─ frontend/
│  │  ├─ app.py ⭐
│  │  ├─ main_page.py
│  │  ├─ sidebar.py
│  │  ├─ support.py
│  │  ├─ test.py
│  │  ├─ __init__.py
│  │  └─ pages/
│  │     ├─ __init__.py
│  │     ├─ 1_dataset_selection.py ⭐
│  │     ├─ 2_optimization.py ⭐
│  │     ├─ 3_shap_analysis.py ⭐
│  │     └─ shap.py
│  │
│  └─ backend/
│     ├─ __init__.py [exports: Optimizer, DataPreparation, XAI, create_model]
│     ├─ models.py
│     ├─ data.py
│     ├─ tuners/
│     │  ├─ __init__.py
│     │  └─ optimizer.py
│     ├─ utils/
│     │  ├─ data_utils/
│     │  │  ├─ prepare.py [DataPreparation class]
│     │  │  └─ data.py [mock_csv_data, preprocess_data]
│     │  └─ ...
│     └─ xai/
│        ├─ __init__.py
│        ├─ xai.py [XAI, XAIConfig classes]
│        └─ constants.py
│
├─ .streamlit/
│  └─ config.toml [theme configuration]
│
├─ FRONTEND_ARCHITECTURE.md [Documentation]
├─ FRONTEND_QUICK_REFERENCE.md [Quick guide]
└─ FRONTEND_ARCHITECTURE_DIAGRAM.md [This file]

RUNTIME DIRECTORIES (created at runtime):
│
├─ uploaded_data/ [Temporary CSV uploads from file_uploader]
│  └─ *.csv
│
├─ data/ [Processed datasets from Page 1]
│  └─ {dataset_name}.csv
│
├─ db/ [Optuna SQLite databases from Page 2]
│  └─ {db_name}.db
│
├─ outputs/ [User-saved SHAP plots from Page 3]
│  └─ *.png
│
└─ .streamlit/
   ├─ .streamlit_cache/ [Streamlit cache]
   └─ logs/ [Streamlit logs]
```

---

## 8. Control Flow Diagram (Swimlanes)

```
User          │    Frontend      │    Backend       │    Storage
              │                  │                  │
Load Dataset  │                  │                  │
  ├─→ Page 1 ─┼─→ fetch_uci() ──┼─→ ucimlrepo API │
  │           │    (or upload)   │                  │
  │           │                  │                  │
  └─→ Configure
      Target ─┼─→ Validate data  ├─→ mock_csv_data() ─→ data/*.csv
              │    Transformation │                  │
              │                  │                  │
Prepare Data  │                  │                  │
  ├─→ Page 2 ─┼──────────────────┼─→ DataPrep  ────┤
  │           │                  │  - Read CSV     │
  │           │                  │  - Scale        │
  │           │  (data_dict)      │  - Split        │
  └─→ Optimize│◄──────────────────┤  - Map target   │
              │                  │                  │
              │                  │                  │
              │  Trial 1          │                  │
              │  ├─→ Model create ├─→ create_model()│
Run N Trials  │  ├─→ Train        │  ├─→ model.fit()│
              │  ├─→ Evaluate     │  ├─→ f1_score() │
              │  ├─→ Store metrics├──┼─→ trial.user_attr
              │  └─→ ...N trials  │                  │
              │                  │                  ├─→ db/*.db
              │  Best results ◄──┴─→ study.best_trials
              │                  │                  │
Analyze Model │                  │                  │
  ├─→ Page 3 ─┼─→ Train model ──┤─→ create_model()│
  │           │                  │  + model.fit()  │
  │           │                  │                  │
  │           │─→ SHAP analysis ─┼─→ XAI class ────┤
  │           │  (calculate      │  ├─→ Explainer() │
  │           │   values)         │  ├─→ get_plot()│
  │           │                  │  └─→ metrics    │
  │           │                  │                  ├─→ outputs/*.png
  │           │                  │                  │
  │           │─→ Generate report├─→ LangChain  ───┤
  │           │  (with LLM)      │  ├─→ ChatGPT/  │
  └─→ Download│  ◄─ Markdown    │     Gemini/    │
     Report   │    Report        │     Claude     │
              │  (*.md file)     │                  │
```

---

## 9. Key Metrics & Performance Path

```
Optimization Trial Loop:
┌─────────────────────────────────────────────────────┐
│ for trial in 1..n_trials:                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Suggest Hyperparameters (TPE Sampler)          │
│     ├─ max_vmap: [1]                              │
│     ├─ batch_size: [32]                           │
│     ├─ learning_rate: [0.001, 0.01, 0.1]         │
│     ├─ n_layers: [1, 5, 10]                       │
│     ├─ model_type: [15 options]                   │
│     └─ [20+ other params]                          │
│                                                     │
│  2. Create Model                                   │
│     └─ create_model(model_type, **params)         │
│                                                     │
│  3. Train Model                                    │
│     └─ model.fit(train_x, train_y)                │
│        └─ Training time: 1-30 seconds (varies)    │
│                                                     │
│  4. Evaluate Model                                 │
│     ├─ predictions = model.predict(test_x)        │
│     ├─ f1 = f1_score(test_y, predictions)         │
│     ├─ accuracy = accuracy_score(...)             │
│     └─ Return f1 as objective value                │
│                                                     │
│  5. Log Metrics                                    │
│     ├─ trial.set_user_attr("Quantum_f1_score", f1)│
│     ├─ trial.set_user_attr("Classical_f1_score",0)│
│     ├─ trial.set_user_attr("Quantum_accuracy", ...)
│     └─ trial.set_user_attr("Classical_accuracy",0)│
│                                                     │
│  6. Persist to SQLite                              │
│     └─ Storage: db/{db_name}.db                   │
│                                                     │
└─────────────────────────────────────────────────────┘
       │
       │ After N trials
       ↓
┌─────────────────────────────────────────────────────┐
│ Best Trials Selection                               │
├─────────────────────────────────────────────────────┤
│                                                     │
│ study.best_trials:                                 │
│ ├─ Sorted by F1 score (descending)                │
│ ├─ Top 5 displayed in results                      │
│ └─ Selected for SHAP analysis                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 10. Technology Stack Summary

```
FRONTEND FRAMEWORK:
├─ Streamlit 1.x
│  ├─ Multi-page app (st.set_page_config)
│  ├─ Session state persistence
│  ├─ Responsive UI components
│  ├─ Tabs, expanders, columns
│  ├─ File uploaders
│  ├─ Chart rendering (native + plotly)
│  └─ CSS/HTML customization

CORE ML LIBRARIES:
├─ scikit-learn
│  ├─ Classification models (SVC, MLP, Perceptron)
│  ├─ Metrics (F1, precision, recall, AUC, etc.)
│  ├─ Preprocessing (StandardScaler, train_test_split)
│  └─ Confusion matrix

├─ PennyLane
│  ├─ Quantum circuit definition
│  ├─ 18 quantum model types
│  ├─ Quantum simulators
│  └─ Hybrid quantum-classical training

├─ Optuna
│  ├─ Hyperparameter optimization
│  ├─ TPE sampler
│  ├─ Study management
│  └─ SQLite backend

DATA HANDLING:
├─ pandas
│  ├─ DataFrame operations
│  ├─ CSV I/O
│  └─ Data transformation

├─ numpy
│  ├─ Array operations
│  ├─ Numerical computations
│  └─ Data storage (train/test splits)

EXPLAINABILITY:
├─ SHAP
│  ├─ SHAP Explainer
│  ├─ Feature importance calculation
│  ├─ 5 plot types (bar, beeswarm, violin, heatmap, waterfall)
│  └─ Confusion matrix visualization

AI/LLM INTEGRATION:
├─ LangChain
│  ├─ Chat model abstractions
│  ├─ Prompt templating
│  ├─ Multimodal message handling
│  └─ Chain orchestration

├─ langchain-google-genai
│  ├─ Google Generative AI (Gemini)
│  └─ Vision capability for images

├─ langchain-openai
│  ├─ OpenAI API integration
│  ├─ GPT-4, GPT-4V (vision)
│  └─ Embedding support

DATA SOURCE:
├─ ucimlrepo
│  ├─ UCI ML Repository API
│  ├─ 400+ datasets
│  └─ Metadata retrieval

VISUALIZATION:
├─ matplotlib
│  ├─ SHAP plot rendering
│  ├─ Figure saving
│  └─ Confusion matrix plots

├─ plotly
│  ├─ Interactive charts
│  ├─ Optuna visualizations
│  └─ Timeline, parameter importance, history

STORAGE:
├─ SQLite 3
│  ├─ Optuna trial database
│  ├─ db/{db_name}.db
│  └─ ACID compliance

├─ File System
│  ├─ CSV files (data/)
│  ├─ PNG outputs (outputs/)
│  └─ Markdown reports (download)
```

---

## Summary Architecture Points

1. **Multi-Page Design:** Each page is a self-contained Streamlit app with its own workflow
2. **Session-Based State:** Streamlit session_state maintains data across pages
3. **Backend Separation:** Clean separation between frontend UI and backend logic
4. **Factory Pattern:** Models created via `create_model()` factory function
5. **Lazy Loading:** SHAP Explainer and values computed on-demand
6. **Modular Libraries:** Each external library handles one responsibility
7. **File-Based Persistence:** SQLite for trials, CSV for prepared data
8. **API Abstraction:** LangChain abstracts multiple LLM providers
9. **Error Handling:** Try/except blocks with user-friendly error messages
10. **Progressive Disclosure:** Information revealed step-by-step through pages
