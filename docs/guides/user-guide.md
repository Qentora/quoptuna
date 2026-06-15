# User Guide

## Introduction

QuOptuna is a comprehensive platform for quantum-enhanced machine learning optimization. This guide will walk you through the complete workflow from dataset selection to model analysis and report generation.

## Workflow Overview

The QuOptuna workflow consists of four main stages:

1. **Dataset Selection** - Load and prepare your data
2. **Optimization** - Find the best hyperparameters
3. **Model Training** - Train models with optimized parameters
4. **SHAP Analysis** - Understand and explain model behavior

## Getting Started

### Installation

Install QuOptuna using UV (recommended) or pip:

```bash
# Using UV (recommended)
uv pip install quoptuna

# Using pip
pip install quoptuna
```

### Launching the Application

Start the Streamlit interface:

```bash
quoptuna --start
```

Or using Python:

```bash
python -m quoptuna.frontend.app run
```

## Dataset Selection

### UCI ML Repository

QuOptuna provides easy access to datasets from the UCI Machine Learning Repository:

1. Navigate to the **Dataset Selection** page
2. Select **UCI ML Repository** tab
3. Choose from popular datasets or enter a custom UCI ID
4. Click **Load UCI Dataset**

**Popular Datasets:**
- Statlog (Australian Credit Approval) - ID: 143
- Blood Transfusion Service Center - ID: 176
- Banknote Authentication - ID: 267
- Heart Disease - ID: 45
- Ionosphere - ID: 225

### Custom Dataset Upload

To use your own dataset:

1. Navigate to the **Upload Custom Dataset** tab
2. Upload a CSV file
3. Configure target and feature columns
4. Apply target transformation if needed

### Data Configuration

**Important:** QuOptuna requires binary classification targets to be encoded as `-1` and `1`.

1. **Select Target Column**: Choose the column you want to predict
2. **Select Features**: Choose the features to use for prediction
3. **Target Transformation**: Map your target values to -1 and 1
4. **Handle Missing Values**: QuOptuna will automatically remove rows with missing values

Click **Save Configuration** to proceed to the next step.

## Data Preparation & Optimization

### Data Preparation

Once your dataset is configured:

1. Review the dataset summary (rows, columns, target distribution)
2. Click **Prepare Data for Training**
3. QuOptuna will automatically:
   - Split data into training and test sets
   - Scale features
   - Convert to the format required by models

### Hyperparameter Optimization

Configure and run optimization:

1. **Database Name**: Name for storing optimization results
2. **Study Name**: Unique identifier for this optimization study
3. **Number of Trials**: How many hyperparameter combinations to try (recommended: 50-200)

Click **Start Optimization** to begin. This will:
- Test multiple model types (both quantum and classical)
- Try different hyperparameter combinations
- Track the best performing configurations

**Model Types Tested:**
- Data Reuploading Classifier (Quantum)
- Circuit-Centric Classifier (Quantum)
- Quantum Kitchen Sinks (Quantum)
- Support Vector Classifier (Classical)
- Multi-Layer Perceptron (Classical)
- And more...

### Understanding Results

After optimization completes, you'll see:
- **Best Trials**: Top performing configurations
- **F1 Scores**: Performance metrics for quantum and classical approaches
- **Hyperparameters**: The configuration for each trial

## SHAP Analysis & Reporting

### Trial Selection

1. Navigate to the **SHAP Analysis** page
2. Select a trial from the dropdown (sorted by performance)
3. Review the trial details and parameters

### Model Training

1. Click **Train Model** to train the selected model
2. The model will be trained on your data with the optimized hyperparameters

### SHAP Analysis

Configure SHAP analysis:

- **Use Probability Predictions**: Use probability outputs instead of class predictions
- **Use Subset of Data**: Analyze a subset for faster computation
- **Subset Size**: Number of samples to analyze (recommended: 50-100)

Click **Run SHAP Analysis** to calculate SHAP values.

### SHAP Visualizations

QuOptuna provides multiple visualization types:

#### Bar Plot
Shows the mean absolute SHAP value for each feature, indicating overall importance.

**Use Case:** Quick overview of feature importance

#### Beeswarm Plot
Shows the distribution of SHAP values, with color indicating feature value (red = high, blue = low).

**Use Case:** Understanding how feature values affect predictions

#### Violin Plot
Shows the distribution of SHAP values for each feature.

**Use Case:** Understanding the variability in feature impact

#### Heatmap
Shows SHAP values for individual instances.

**Use Case:** Instance-level analysis, finding patterns in predictions

#### Waterfall Plot
Explains how features contribute to a single prediction.

**Use Case:** Understanding individual predictions in detail

#### Confusion Matrix
Shows classification performance.

**Use Case:** Evaluating overall model accuracy

### Report Generation

Generate comprehensive AI-powered reports:

1. **Select LLM Provider**: Google (Gemini), OpenAI (GPT), or Anthropic (Claude)
2. **Enter API Key**: Your API key for the selected provider
3. **Model Name**: Specific model to use (e.g., "models/gemini-2.0-flash-exp")
4. **Dataset Information** (optional): Add context about your dataset

Click **Generate Report** to create a detailed analysis report.

**Report Includes:**
- Performance metrics analysis
- SHAP value interpretation
- Feature importance ranking
- Risk and fairness assessment
- Governance recommendations

## Best Practices

### Optimization

- **Start Small**: Begin with 50-100 trials to get quick results
- **Increase Gradually**: Use 100-200 trials for production models
- **Monitor Performance**: Check both quantum and classical model scores
- **Save Studies**: Use descriptive names for databases and studies

### SHAP Analysis

- **Use Subsets**: Analyze 50-100 samples for faster computation
- **Multiple Plots**: Generate several plot types for comprehensive understanding
- **Document Findings**: Save plots and reports for future reference
- **Understand Context**: Consider domain knowledge when interpreting SHAP values

### Report Generation

- **Provide Context**: Add dataset URL and description for better AI insights
- **Choose Appropriate Models**:
  - Fast models (Gemini Flash): Quick exploratory reports
  - Advanced models (GPT-4, Gemini Pro): Detailed production reports
- **Review Carefully**: AI-generated reports should be reviewed by domain experts

## Troubleshooting

### Common Issues

**Dataset Loading Fails**
- Check UCI dataset ID is correct
- Ensure CSV file is properly formatted
- Verify file encoding (UTF-8 recommended)

**Optimization Errors**
- Ensure data has no missing values
- Check target column has exactly 2 unique values
- Verify sufficient samples for train/test split

**SHAP Analysis Slow**
- Reduce subset size
- Use simpler model types
- Check available memory

**Report Generation Fails**
- Verify API key is valid
- Check internet connection
- Ensure model name is correct
- Try a different LLM provider

## Advanced Features

### Loading Previous Studies

You can load and analyze previously run optimizations:

1. Go to the **Optimization** page
2. Enter the database name and study name
3. Click **Load Optimizer**
4. Results will be available for analysis

### Batch Processing

For multiple datasets, you can:
1. Use the Python API directly (see API documentation)
2. Script the workflow using QuOptuna classes
3. Save results to different databases

### Custom Models

Advanced users can integrate custom models by:
1. Following the model interface in `quoptuna.backend.models`
2. Adding model configurations to the optimizer
3. See API documentation for details

## Next Steps

- Explore the [API Documentation](api_reference.md) for programmatic usage
- Check out [Examples](examples.md) for common use cases
- Contribute on [GitHub](https://github.com/Qentora/quoptuna)

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Qentora/quoptuna/issues)
- **Documentation**: [Full documentation](https://qentora.github.io/quoptuna)
- **Community**: Join our discussions on GitHub
