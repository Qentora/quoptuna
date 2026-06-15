# QuOptuna Streamlit Interface Guide

## Overview

The QuOptuna Streamlit interface provides a comprehensive, user-friendly workflow for quantum-enhanced machine learning optimization. This guide covers the enhanced multi-page application structure and features.

## New Features

### 🎨 Multi-Page Application

The interface is now organized into dedicated pages for each workflow stage:

1. **Home** - Welcome page with quick start guide
2. **📊 Dataset Selection** - Load and configure datasets
3. **🎯 Optimization** - Data preparation and hyperparameter optimization
4. **🔍 SHAP Analysis** - Model training and explainable AI

### 📊 Dataset Selection Page

**Location:** `src/quoptuna/frontend/pages/1_dataset_selection.py`

#### Features:

**UCI ML Repository Integration**
- Popular datasets pre-configured (Statlog, Blood, Banknote, Heart Disease, Ionosphere)
- Custom UCI ID support for accessing any UCI dataset
- Automatic metadata display (instances, features, description)

**Custom Dataset Upload**
- CSV file upload support
- Automatic data preview
- Missing value detection

**Data Configuration**
- Interactive column selection for features and target
- Target transformation to -1/+1 encoding
- Automatic preprocessing (missing value removal)
- Data validation before proceeding

**Example Usage:**
```python
# Programmatically access the same functionality
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=143)  # Statlog dataset
X = dataset.data.features
y = dataset.data.targets
```

### 🎯 Optimization Page

**Location:** `src/quoptuna/frontend/pages/2_optimization.py`

#### Features:

**Data Preparation**
- Automatic train/test splitting
- Feature scaling
- Numpy array conversion for model compatibility
- Dataset summary statistics

**Hyperparameter Optimization**
- Configurable number of trials (10-200)
- Custom database and study names
- Progress tracking
- Real-time status updates

**Results Visualization**
- Best trial ranking
- Performance metrics (Quantum F1, Classical F1)
- Parameter inspection
- Model type comparison

**Example Usage:**
```python
from quoptuna import DataPreparation, Optimizer

# Prepare data
data_prep = DataPreparation(file_path="data/my_data.csv", x_cols=features, y_col="target")
data_dict = data_prep.get_data(output_type="2")

# Run optimization
optimizer = Optimizer(db_name="my_exp", study_name="trial_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)
```

### 🔍 SHAP Analysis Page

**Location:** `src/quoptuna/frontend/pages/3_shap_analysis.py`

#### Features:

**Trial Selection**
- Dropdown with formatted trial information
- Performance metrics display
- Full parameter inspection

**Model Training**
- One-click model training with optimized parameters
- Training status feedback
- Model storage in session state

**SHAP Configuration**
- Probability vs class prediction toggle
- Subset analysis option
- Configurable subset size

**Visualization Suite**
Six types of SHAP visualizations:

1. **Bar Plot** - Overall feature importance
2. **Beeswarm Plot** - Feature value impact distribution
3. **Violin Plot** - SHAP value distributions
4. **Heatmap** - Instance-level contributions
5. **Waterfall Plot** - Individual prediction explanations
6. **Confusion Matrix** - Model performance

**AI Report Generation**
- Support for multiple LLM providers (Google, OpenAI, Anthropic)
- Configurable model selection
- Optional dataset context
- Markdown export

**Example Usage:**
```python
from quoptuna import XAI
from quoptuna.backend.xai.xai import XAIConfig

# Configure XAI
config = XAIConfig(use_proba=True, onsubset=True, subset_size=50)
xai = XAI(model=trained_model, data=data_dict, config=config)

# Generate visualizations
bar_plot = xai.get_plot("bar", max_display=10, class_index=1)
beeswarm = xai.get_plot("beeswarm", max_display=10, class_index=1)

# Generate AI report
report = xai.generate_report_with_langchain(
    provider="google",
    api_key="your-api-key",
    model_name="models/gemini-2.0-flash-exp"
)
```

## Application Structure

```
src/quoptuna/frontend/
├── app.py                  # Main application entry point
├── main_page.py            # Home page with quick start guide
├── sidebar.py              # Legacy sidebar (kept for compatibility)
├── support.py              # Helper functions
└── pages/
    ├── 1_dataset_selection.py    # Dataset loading and configuration
    ├── 2_optimization.py         # Data prep and optimization
    └── 3_shap_analysis.py        # SHAP analysis and reporting
```

## Session State Management

The application uses Streamlit's session state to maintain data across pages:

```python
# Key session state variables
st.session_state = {
    # Dataset Selection
    "dataset_loaded": bool,
    "dataset_df": pd.DataFrame,
    "dataset_name": str,
    "file_path": str,
    "target_column": str,
    "feature_columns": list,

    # Optimization
    "data_dict": dict,
    "optimizer": Optimizer,
    "study": optuna.Study,
    "best_trials": list,
    "optimization_complete": bool,

    # SHAP Analysis
    "selected_trial": FrozenTrial,
    "trained_model": Model,
    "xai": XAI,
    "report": str,
}
```

## Workflow Example

### Complete Workflow Through UI

1. **Start Application**
   ```bash
   quoptuna --start
   ```

2. **Dataset Selection**
   - Navigate to "📊 Dataset Selection"
   - Select "UCI ML Repository" tab
   - Choose "Statlog (Australian Credit Approval)"
   - Click "Load UCI Dataset"
   - Review metadata
   - Select target column (will auto-select features)
   - Enable "Transform target values to -1 and 1"
   - Click "Save Configuration"

3. **Optimization**
   - Navigate to "🎯 Optimization"
   - Click "Prepare Data for Training"
   - Review dataset summary
   - Set number of trials (e.g., 100)
   - Enter database name (e.g., "Statlog")
   - Enter study name (e.g., "Statlog_Trial_1")
   - Click "Start Optimization"
   - Wait for completion
   - Review best trials

4. **SHAP Analysis**
   - Navigate to "🔍 SHAP Analysis"
   - Select best trial from dropdown
   - Review trial details
   - Click "Train Model"
   - Configure SHAP settings
   - Click "Run SHAP Analysis"
   - Navigate through visualization tabs
   - (Optional) Generate AI report with LLM

## Customization

### Adding New Pages

Create a new page in `src/quoptuna/frontend/pages/`:

```python
# src/quoptuna/frontend/pages/4_my_feature.py

import streamlit as st

def main():
    st.set_page_config(
        page_title="My Feature - QuOptuna",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 My Feature")
    # Your implementation here

if __name__ == "__main__":
    main()
```

Streamlit automatically detects and adds pages to the sidebar.

### Modifying Visualizations

Edit `src/quoptuna/frontend/pages/3_shap_analysis.py`:

```python
def display_shap_plots():
    # Add custom visualization
    with tabs[6]:  # Add new tab
        st.markdown("### Custom Visualization")
        # Your custom plot code
```

### Customizing Styles

Edit `src/quoptuna/frontend/app.py`:

```python
st.markdown(
    """
    <style>
    /* Your custom CSS */
    .main-title {
        color: #your-color;
    }
    </style>
    """,
    unsafe_allow_html=True
)
```

## Best Practices

### Performance

1. **Use Data Subsets for SHAP**
   - Limit subset size to 50-100 samples for faster computation
   - Use full dataset only when necessary

2. **Optimize Trial Count**
   - Start with 50-100 trials for initial exploration
   - Increase to 100-200 for production models

3. **Cache Results**
   - Session state persists data during the session
   - Database stores optimization results permanently

### User Experience

1. **Clear Navigation**
   - Use descriptive page titles
   - Provide progress indicators
   - Show clear error messages

2. **Data Validation**
   - Validate inputs before processing
   - Provide helpful warning messages
   - Guide users to correct issues

3. **Documentation**
   - Use expanders for detailed information
   - Provide tooltips for parameters
   - Link to external documentation

## Troubleshooting

### Common Issues

**"Please complete Dataset Selection first"**
- Ensure you've saved the dataset configuration on the Dataset Selection page
- Check that `file_path` is in session state

**"Please prepare data first"**
- Click "Prepare Data for Training" on the Optimization page
- Verify `data_dict` is populated in session state

**"Please run SHAP analysis first"**
- Ensure model is trained
- Click "Run SHAP Analysis" button
- Check for error messages

**Plots Not Displaying**
- Check internet connection (for base64 encoded images)
- Verify SHAP calculation completed successfully
- Try reducing subset size

**Report Generation Fails**
- Verify API key is valid and active
- Check model name format (e.g., "models/gemini-2.0-flash-exp")
- Ensure internet connection is stable
- Try a different LLM provider

### Debug Mode

Enable debug information:

```python
import streamlit as st

# Add to any page
with st.expander("Debug Info"):
    st.write("Session State:", st.session_state)
```

## API Integration

The Streamlit interface can be used alongside the Python API:

```python
# Load data from Streamlit session
import streamlit as st
from quoptuna import Optimizer

# Use optimizer created in UI
if "optimizer" in st.session_state:
    optimizer = st.session_state["optimizer"]
    study = optimizer.study

    # Continue with Python API
    for trial in study.best_trials:
        print(trial.params)
```

## Contributing

To contribute to the Streamlit interface:

1. Fork the repository
2. Create a feature branch
3. Add/modify pages in `src/quoptuna/frontend/pages/`
4. Test thoroughly with different datasets
5. Update this documentation
6. Submit a pull request

## Support

For issues or questions:
- **GitHub Issues**: [Report bugs](https://github.com/Qentora/quoptuna/issues)
- **Documentation**: [Full docs](https://qentora.github.io/quoptuna)
- **Examples**: See `docs/examples.md`

## License

MIT License - See LICENSE file for details
