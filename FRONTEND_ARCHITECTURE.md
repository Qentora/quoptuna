# QuOptuna Frontend Architecture - Comprehensive Overview

## 1. Project Structure

The QuOptuna frontend is built with **Streamlit** and follows a modular, multi-page architecture.

### Directory Layout
```
src/quoptuna/frontend/
├── app.py                      # Main entry point & page configuration
├── main_page.py               # Welcome/home page
├── sidebar.py                 # Sidebar with data upload & database configuration
├── support.py                 # Shared utilities & visualization helpers
└── pages/
    ├── 1_dataset_selection.py # Dataset loading & preparation
    ├── 2_optimization.py      # Hyperparameter optimization
    ├── 3_shap_analysis.py     # SHAP analysis & report generation
    └── shap.py               # Legacy SHAP utilities
```

---

## 2. Pages/Components Overview

### 2.1 Main Application Page (app.py)
**Purpose:** Application entry point and page configuration

**Features:**
- Initializes Streamlit page config (wide layout, atom icon)
- Applies custom CSS styling (dark theme with purple accents)
- Initializes session state
- Coordinates sidebar and main page rendering
- Manages visualization updates

**Key Components:**
```python
st.set_page_config(page_title="QuOptuna", page_icon=":atom_symbol:", layout="wide")
# Purple theme: #9b59b6
# Dark background: #121212
```

---

### 2.2 Main/Home Page (main_page.py)
**Purpose:** Welcome page with overview and getting started guide

**Displays:**
- Welcome heading with project description
- 3-column feature highlights:
  - Optimization capabilities
  - Explainability features (SHAP)
  - Analytics & reporting
- Step-by-step workflow guide
- Tips & best practices for users
- Links to GitHub & issue tracking

**Supported Models Advertised:**
- **Quantum Models (7+):**
  - Data Reuploading Classifier
  - Circuit-Centric Classifier
  - Quantum Kitchen Sinks
  - Quantum Metric Learner
  - Dressed Quantum Circuit Classifier
  - And others...
  
- **Classical Models:**
  - Support Vector Classifier (SVC)
  - Multi-Layer Perceptron (MLP)
  - Perceptron

---

### 2.3 Page 1: Dataset Selection (pages/1_dataset_selection.py)
**Purpose:** Load and prepare datasets for optimization

**Two Data Sources:**
1. **UCI ML Repository** (via `ucimlrepo` library)
   - Popular datasets with pre-configured IDs
   - Options: Australian Credit, Blood Transfusion, Banknote Auth, Heart Disease, Ionosphere
   - Custom dataset ID support
   - Displays metadata (instances, features, area, tasks)

2. **Custom CSV Upload**
   - File uploader for user datasets
   - Validates CSV format

**Configuration Features:**
- Data preview with statistics
- Target & feature column selection
- Target value transformation to -1 and 1 (binary classification requirement)
- Missing value handling (removal)
- Column renaming (target → "target")
- Data saved to `data/` directory via `mock_csv_data()` utility

**Session State Tracking:**
- `dataset_df`: Loaded DataFrame
- `file_path`: Path to saved processed data
- `target_column`, `feature_columns`: Selected columns
- `dataset_name`: Dataset identifier

---

### 2.4 Page 2: Optimization (pages/2_optimization.py)
**Purpose:** Data preparation and hyperparameter optimization

**Two-Stage Workflow:**

#### Stage 1: Data Preparation
- Uses `DataPreparation` class
- Reads prepared CSV from Dataset Selection page
- Performs train/test split (75/25 ratio)
- StandardScaler normalization
- Returns data dict with keys: `train_x`, `test_x`, `train_y`, `test_y`
- Shows train/test split metrics

#### Stage 2: Optimization
- Uses `Optimizer` class from backend
- Configuration inputs:
  - Database name
  - Study name
  - Number of trials (10-200 range)
- Runs hyperparameter tuning with Optuna
- Displays progress bar
- Shows best trials with metrics:
  - Trial number
  - Model type
  - Quantum F1 Score
  - Classical F1 Score
  - Key parameters (learning_rate, n_layers, batch_size, C, gamma)

**Key Optimized Hyperparameters:**
- Model selection from 15 model types
- Learning rates: [0.001, 0.01, 0.1]
- Layers: [1, 5, 10]
- Batch sizes, C values, gamma factors
- Quantum-specific: observable type, n_qfeatures, n_episodes, etc.

**Session State:**
- `data_dict`: Prepared training/test data
- `optimizer`: Optimizer instance
- `study`: Optuna study object
- `best_trials`: Top performing trials
- `optimization_complete`: Boolean flag

---

### 2.5 Page 3: SHAP Analysis (pages/3_shap_analysis.py)
**Purpose:** Model explainability with SHAP and AI-powered reporting

**Four-Stage Workflow:**

#### Stage 1: Trial Selection
- Dropdown to select from best trials
- Displays trial details, performance metrics, all parameters
- Session state: `selected_trial`

#### Stage 2: Model Training
- Instantiates model with selected trial's hyperparameters
- Trains on full training dataset
- Session state: `trained_model`

#### Stage 3: SHAP Analysis
- Configurable SHAP parameters:
  - Use probability predictions vs class predictions
  - Use data subset vs full dataset
  - Subset size slider (10-200 samples)
- Creates `XAI` instance with model and config
- Session state: `xai`

#### Stage 4: Visualizations & Report
- **5 SHAP Plot Types** (in tabs):
  1. **Bar Plot** - Feature importance ranking
  2. **Beeswarm Plot** - Feature impact distribution
  3. **Violin Plot** - SHAP value distributions
  4. **Heatmap** - Instance-level analysis (up to 50 instances)
  5. **Waterfall Plot** - Individual prediction explanation

- **Additional Visualization:**
  6. Confusion Matrix - Model classification performance

- **AI Report Generation:**
  - LLM provider selection: Google, OpenAI, Anthropic
  - Model name customization
  - Dataset context (optional):
    - URL, description, features, target
  - Report configuration with API key input
  - LangChain integration for multimodal report generation
  - Markdown report download

**Plot Features:**
- All plots saved as base64-encoded images
- Save individual plots to `outputs/` directory
- Max display config for feature limiting

---

## 3. Backend APIs and Services

### 3.1 Core Backend Modules

#### `quoptuna.Optimizer`
**Location:** `backend/tuners/optimizer.py`

**Responsibilities:**
- Hyperparameter optimization with Optuna
- SQLite database management (`db/` folder)
- Trial execution and metric logging

**Key Methods:**
- `optimize(n_trials)` - Run optimization loop
- `objective(trial)` - Objective function for each trial
- `log_user_attributes()` - Store quantum vs classical metrics
- `load_study()` - Load existing optimization study

**Data Flow:**
1. Receives train/test data (numpy arrays)
2. Creates Optuna study with TPE sampler
3. Iteratively:
   - Suggests hyperparameters
   - Creates model with `create_model()`
   - Trains and evaluates
   - Records F1 scores, accuracy
   - Stores metrics in trial user_attrs

**Storage:** SQLite at `sqlite:///db/{db_name}.db`

---

#### `quoptuna.DataPreparation`
**Location:** `backend/utils/data_utils/prepare.py`

**Responsibilities:**
- Load and normalize data
- Train/test splitting
- Feature scaling

**Key Methods:**
- `__init__(file_path, x_cols, y_col)` - Load from CSV
- `prepare_data()` - Full preprocessing pipeline
- `preprocess(x, y)` - Scaling and train/test split
- `get_data(output_type)` - Return formatted data dict

**Processing Pipeline:**
1. Read CSV file
2. Select feature & target columns
3. StandardScaler fit on training data
4. Train/test split (75/25, random_state=42)
5. Transform target: unique classes → {-1, 1}

---

#### `quoptuna.create_model(model_type, **kwargs)`
**Location:** `backend/models.py`

**Supports 26 Models:**
- **Quantum (18):** CircuitCentric, DataReuploading, DressedQuantumCircuit, etc.
- **Classical (8):** SVC, MLPClassifier, Perceptron, etc.

**Factory Pattern:**
```python
create_model(
    model_type="DataReuploadingClassifier",
    max_vmap=1,
    batch_size=32,
    learning_rate=0.01,
    n_layers=5,
    observable_type="full"
)
```

---

#### `quoptuna.XAI` (Explainability)
**Location:** `backend/xai/xai.py`

**Responsibilities:**
- SHAP value calculation and visualization
- Model performance metrics
- AI-powered report generation

**Key Features:**

**SHAP Methods:**
- `get_plot(type, class_index, index)` - Generate plot (bar, beeswarm, violin, heatmap, waterfall)
- `get_shap_values` - Calculate SHAP values for data
- `explainer` - SHAP Explainer instance (lazy-loaded)

**Metrics Methods:**
- Classification: precision, recall, F1, MCC, Cohen's Kappa
- ROC/AUC: roc_curve, roc_auc_score
- Confusion matrix & classification report
- Log loss, average precision

**Report Generation:**
- `generate_report_with_langchain(api_key, model_name, provider)`
- LLM providers: Google GenAI, OpenAI
- Multimodal prompt with base64-encoded images
- Configurable system prompt from `prompt.txt`

**Config:**
```python
XAIConfig(
    use_proba=True,           # Use probability predictions
    onsubset=True,            # Use data subset
    subset_size=100,          # Subset size
    max_display=20,           # Max features to show
    feature_names=None        # Auto-detect from data
)
```

---

### 3.2 Data Flow Diagram

```
USER INPUT
    ↓
┌─────────────────────────────────────┐
│ Page 1: Dataset Selection           │
│ - Load from UCI or CSV upload       │
│ - Select target & features          │
│ - Transform binary target (-1, 1)   │
│ - Save to data/ folder              │
└─────────────────────────────────────┘
    ↓ (file_path)
┌─────────────────────────────────────┐
│ Page 2: Optimization                │
│ - DataPreparation loads & splits    │
│ - Optimizer.optimize() runs trials  │
│ - Tests 15 model types              │
│ - Records quantum/classical scores  │
│ - Stores in SQLite DB               │
└─────────────────────────────────────┘
    ↓ (best_trials)
┌─────────────────────────────────────┐
│ Page 3: SHAP Analysis               │
│ - Train model with best params      │
│ - XAI.get_shap_values() calculate   │
│ - 5 visualization types             │
│ - LangChain generate AI report      │
│ - Download markdown report          │
└─────────────────────────────────────┘
```

---

## 4. Communication & External Services

### 4.1 Backend API Calls
**Type:** In-process Python library calls (no HTTP API)

All backend functionality accessed via direct Python imports:
```python
from quoptuna import Optimizer, DataPreparation, XAI, create_model
```

### 4.2 External Services Used

#### UCI ML Repository
- **Library:** `ucimlrepo`
- **Purpose:** Download datasets
- **Datasets Available:** 400+
- **Integration:** Pages 1 - fetch_uci_dataset()

#### Optuna
- **Purpose:** Hyperparameter optimization framework
- **Storage:** SQLite database
- **Sampler:** Tree-structured Parzen Estimator (TPE)
- **Integration:** Page 2 - Optimizer class

#### LangChain
- **Purpose:** LLM integration for report generation
- **Providers Supported:**
  - Google GenAI (Gemini models)
  - OpenAI (GPT models)
  - Anthropic (Claude models)
- **Features:** Multimodal prompting with images
- **Integration:** Page 3 - generate_report_with_langchain()

#### SHAP
- **Purpose:** Model explainability library
- **Features:** Explainer, plot generation
- **Visualizations:** bar, beeswarm, waterfall, violin, heatmap
- **Integration:** Page 3 - XAI class

#### Scikit-learn
- **Purpose:** Classical ML models & metrics
- **Models:** SVC, MLPClassifier, Perceptron
- **Metrics:** F1, precision, recall, confusion matrix
- **Integration:** Page 2 (optimization), Page 3 (evaluation)

#### PennyLane
- **Purpose:** Quantum ML framework
- **Models:** Quantum circuit classifiers
- **Integration:** Page 2 (optimization) - 18 quantum model types

---

## 5. Session State Management

### Session State Keys
```python
# Sidebar
uploaded_file                  # Uploaded CSV file object
uploaded_file_name            # Original filename
file_location                 # Path to uploaded file
x_columns, y_column          # Selected columns
DB_NAME, study_name          # Optuna configuration
n_trials                      # Number of trials
optimizer                     # Optimizer instance
process_running               # Background optimization flag
start_visualization           # Visualization start flag

# Dataset Selection Page
dataset_loaded                # Boolean flag
dataset_df                    # DataFrame
dataset_name                  # Display name
dataset_metadata              # UCI metadata
file_path                     # Processed data path
target_column, feature_columns # Selected columns

# Optimization Page
data_dict                     # {train_x, test_x, train_y, test_y}
study                         # Optuna study object
best_trials                   # List of best trials
optimization_complete         # Boolean flag
db_name, study_name          # Study identifiers

# SHAP Page
selected_trial                # Selected trial from dropdown
trained_model                 # Trained model instance
xai                          # XAI instance
report                       # Generated markdown report
shap_images                  # Dict of base64-encoded plots
```

---

## 6. Data Formats & Transformations

### Input Data Format (CSV)
```
feature_1, feature_2, ..., feature_n, target_column
0.5,       0.3,       ..., 0.8,       1
...
```

### Processed Data Format (In-Memory Dict)
```python
{
    "train_x": ndarray (n_samples, n_features),
    "test_x": ndarray (m_samples, n_features),
    "train_y": ndarray (n_samples,) with values {-1, 1},
    "test_y": ndarray (m_samples,) with values {-1, 1}
}
```

### Trial Storage Format
```python
Trial:
  number: int
  state: TrialState
  value: float (F1 score)
  params: dict (hyperparameters)
  user_attrs: {
    "Quantum_f1_score": float,
    "Classical_f1_score": float,
    "Quantum_accuracy": float,
    "Classical_accuracy": float,
    ...
  }
  datetime_start, datetime_complete: timestamps
```

### Report Format
Markdown with:
- Model & dataset information
- Performance metrics (F1, precision, recall, etc.)
- SHAP visualizations (base64-encoded PNG)
- AI-generated interpretation of results

---

## 7. Key Features & Capabilities

### Frontend Strengths
✓ **Multi-page Streamlit app** - Clean workflow separation
✓ **Session persistence** - State maintained across pages
✓ **Interactive visualization** - Real-time trial monitoring
✓ **Multiple data sources** - UCI + custom uploads
✓ **Comprehensive ML suite** - 26 model types
✓ **Explainability focus** - 5 SHAP visualization types
✓ **AI reporting** - LLM-powered analysis
✓ **Dark theme** - Purple accent color scheme

### Visualization Capabilities
- **Optuna plots:** Timeline, parameter importance, optimization history
- **SHAP plots:** Bar, beeswarm, violin, heatmap, waterfall
- **Performance:** Confusion matrix, classification report
- **Real-time updates:** 10-second refresh interval during optimization

### Integration Points
- **Database:** SQLite for trial persistence
- **Data source:** UCI ML Repository API
- **LLM APIs:** Google, OpenAI, Anthropic
- **ML frameworks:** Scikit-learn, PennyLane, SHAP

---

## 8. Architecture Patterns

### Design Patterns Used
1. **Factory Pattern** - `create_model(model_type, **kwargs)`
2. **Session State Pattern** - Streamlit `st.session_state`
3. **Configuration Classes** - `XAIConfig` dataclass
4. **Lazy Loading** - Properties in XAI (`@property explainer`)
5. **Strategy Pattern** - Multiple plot types in SHAP
6. **Pipeline Pattern** - Data preparation workflow

### File Organization
- **Pages:** Stateless, self-contained workflows
- **Support:** Shared utilities (`support.py`)
- **Sidebar:** Persistent UI element across pages
- **Backend:** Python library (no web API)

---

## 9. Known Limitations & Edge Cases

### Considerations
- **Binary Classification Only** - Requires target transformation to {-1, 1}
- **Memory Usage** - Full data subsets for SHAP can be memory-intensive
- **Quantum Models** - Requires PennyLane with quantum simulator
- **Optimization Time** - Trials run sequentially, no parallelization in frontend
- **Report Generation** - Requires external LLM API keys
- **Database Concurrency** - SQLite has single-writer limitation

### Error Handling
- Try/except blocks around model training
- Graceful degradation for SHAP visualizations
- API key validation for report generation
- Data validation at each stage

---

## 10. Configuration & Customization

### Streamlit Config (`.streamlit/config.toml`)
```toml
[theme]
base = "dark"
primaryColor = "#8e44ad"        # Purple
backgroundColor = "#121212"     # Dark background
secondaryBackgroundColor = "#1c1c1c"
textColor = "#dcdcdc"           # Light grey
font = "sans serif"
```

### Customizable Parameters
- **Dataset:** Any CSV with binary target
- **Models:** Choice of 26 model types
- **Optimization:** Trial count, database name, study name
- **SHAP:** Subset size, probability vs class mode
- **Reporting:** LLM provider, model name, dataset context

