# QuOptuna Workflow Builder User Guide

## Overview

The QuOptuna Workflow Builder is a visual drag-and-drop interface for creating machine learning pipelines with quantum and classical models. It integrates with the QuOptuna optimization framework to provide hyperparameter tuning and explainability analysis.

## Getting Started

### Running the Application

1. **Start the services:**
   ```bash
   docker compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Creating Your First Workflow

1. Navigate to the **Workflow Builder** page from the sidebar
2. Drag nodes from the **Node Palette** onto the canvas
3. Connect nodes by dragging from one node's output handle to another's input handle
4. Configure each node by clicking on it (future feature)
5. Click **Run** to execute the workflow

## Node Types

### Data Nodes

#### Upload CSV
- **Purpose**: Upload a dataset from a local CSV file
- **Configuration**:
  - Select a CSV file from your computer
  - Backend validates and stores the file
- **Output**: Dataset with columns and row count
- **Usage**: Use this as the starting point for workflows with custom data

#### UCI Dataset
- **Purpose**: Fetch datasets from the UCI Machine Learning Repository
- **Configuration**:
  - Dataset ID (e.g., 53 for Iris dataset)
- **Output**: Dataset loaded from UCI repository
- **Popular Dataset IDs**:
  - 53: Iris
  - 109: Wine Quality
  - 17: Breast Cancer Wisconsin

#### Data Preview
- **Purpose**: View basic statistics and information about your dataset
- **Input**: Connected dataset
- **Output**:
  - Shape (rows, columns)
  - Data types
  - Statistical summary (mean, std, min, max)
  - First few rows

#### Select Features
- **Purpose**: Choose which columns to use as features (X) and target (y)
- **Configuration**:
  - `x_columns`: List of feature column names
  - `y_column`: Target column name
- **Input**: Dataset
- **Output**: Separated X and y data

### Preprocessing Nodes

#### Train/Test Split
- **Purpose**: Split data into training and testing sets
- **Input**: Feature-selected data (X and y)
- **Configuration**:
  - Test size: 0.2 (default, 20% for testing)
  - Random state: 42 (for reproducibility)
- **Output**:
  - `x_train`, `x_test`
  - `y_train`, `y_test`
- **Note**: Automatically handles scaling and encoding via QuOptuna's DataPreparation class

#### Standard Scaler
- **Purpose**: Normalize features to zero mean and unit variance
- **Input**: Split data
- **Output**: Scaled data
- **Note**: Scaling is already handled by DataPreparation, this node is a pass-through for visual clarity

#### Label Encoding
- **Purpose**: Encode categorical target labels as integers
- **Input**: Data with target labels
- **Output**: Encoded labels
- **Note**: Encoding is already handled by DataPreparation, this node is a pass-through for visual clarity

### Model Nodes

#### Quantum Model
- **Purpose**: Configure a quantum machine learning model
- **Configuration**:
  - `model_name`: Choose from 18 quantum models
- **Available Quantum Models**:
  - DataReuploading
  - QuantumKitchen
  - SeparableTwoDesign
  - BasicEntanglerLayers
  - StronglyEntanglingLayers
  - QuantumMetricLearning
  - SimplifiedTwoDesign
  - QCNN
  - TreeTensorNetwork
  - MERA
  - And 8 more...
- **Input**: Preprocessed data
- **Output**: Model configuration for optimization

#### Classical Model
- **Purpose**: Configure a classical machine learning model
- **Configuration**:
  - `model_name`: Choose from 8 classical models
- **Available Classical Models**:
  - RandomForest
  - LogisticRegression
  - SVC
  - GradientBoosting
  - AdaBoost
  - KNN
  - DecisionTree
  - NaiveBayes
- **Input**: Preprocessed data
- **Output**: Model configuration for optimization

### Optimization Nodes

#### Optuna Config
- **Purpose**: Configure hyperparameter optimization settings
- **Configuration**:
  - `study_name`: Name for the optimization study (default: "workflow_study")
  - `n_trials`: Number of optimization trials (default: 100)
  - `db_name`: SQLite database for storing results (default: "workflow_optimization.db")
- **Input**: Model configuration
- **Output**: Optimization parameters merged with model config

#### Run Optimization
- **Purpose**: Execute Optuna hyperparameter optimization
- **Input**: Optuna configuration with model and data
- **Process**:
  1. Creates Optimizer instance with configured parameters
  2. Runs `n_trials` optimization trials
  3. Finds best hyperparameters
  4. Returns best model and performance metrics
- **Output**:
  - `best_value`: Best accuracy/metric achieved
  - `best_params`: Optimal hyperparameters found
  - `best_trial_number`: Which trial was best
  - Study name and database for analysis
- **Duration**: Can take several minutes to hours depending on:
  - Number of trials (more trials = better results but longer time)
  - Model complexity (quantum models are slower)
  - Dataset size

### Analysis Nodes

#### SHAP Analysis
- **Purpose**: Generate explainability visualizations using SHAP (SHapley Additive exPlanations)
- **Input**: Optimization results with trained model
- **Configuration**:
  - `plot_types`: List of plot types to generate
    - "bar": Feature importance bar chart
    - "beeswarm": Feature impact distribution
    - "violin": Feature value distribution
- **Output**: SHAP plots and feature importance rankings
- **Use Case**: Understanding which features impact model predictions

#### Confusion Matrix
- **Purpose**: Visualize classification performance
- **Input**: Model predictions and true labels
- **Output**: Confusion matrix visualization
- **Status**: Placeholder (implementation pending)

#### Feature Importance
- **Purpose**: Rank features by their impact on predictions
- **Input**: Trained model or SHAP results
- **Output**: Feature importance scores
- **Note**: Can use SHAP bar plot if available

### Output Nodes

#### Export Model
- **Purpose**: Save the trained model to disk
- **Configuration**:
  - `export_path`: Where to save the model (default: `./models/{study_name}.pkl`)
- **Input**: Optimization results
- **Output**: Saved model file
- **Status**: Placeholder (implementation pending)

#### Generate Report
- **Purpose**: Create AI-powered analysis report
- **Configuration**:
  - `llm_provider`: "openai", "anthropic", or "google"
- **Input**: Workflow results and metrics
- **Output**: Comprehensive markdown report
- **Status**: Requires LLM API keys to be configured
- **Requirements**: Set environment variables:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`

## Example Workflows

### Simple Classification Workflow

1. **UCI Dataset** → Choose Iris dataset (ID: 53)
2. **Select Features** → X: all except target, y: species
3. **Train/Test Split** → 80/20 split
4. **Quantum Model** → DataReuploading
5. **Optuna Config** → 50 trials
6. **Run Optimization** → Execute training
7. **SHAP Analysis** → Explain predictions

### Upload and Analyze Workflow

1. **Upload CSV** → Your custom dataset
2. **Data Preview** → Inspect the data
3. **Select Features** → Choose relevant columns
4. **Train/Test Split** → Prepare data
5. **Classical Model** → RandomForest
6. **Optuna Config** → 100 trials
7. **Run Optimization** → Find best hyperparameters
8. **Feature Importance** → Understand key features
9. **Export Model** → Save for deployment

### Quantum vs Classical Comparison

Create two parallel paths:

**Path 1 (Quantum):**
- Data → Features → Split → Quantum Model → Optuna → Optimization → SHAP

**Path 2 (Classical):**
- Data → Features → Split → Classical Model → Optuna → Optimization → SHAP

Compare the results to see if quantum models provide advantages for your dataset.

## Workflow Execution

### Execution Process

1. **Topological Sort**: Nodes are sorted into execution order based on dependencies
2. **Sequential Execution**: Each node runs in order, passing data to connected nodes
3. **Background Processing**: Workflows run in the background, allowing you to continue working
4. **Status Updates**: UI polls for updates every 2 seconds
5. **Results**: Final results are displayed when execution completes

### Node Status Indicators

- **Gray**: Not yet executed
- **Blue**: Currently running
- **Green**: Completed successfully
- **Red**: Failed with error

### Monitoring Execution

- Check the status message at the top of the canvas
- Watch node colors change as they execute
- View detailed results in the execution response

## Best Practices

### Data Preparation

1. **Clean Data**: Ensure CSV files are properly formatted
2. **Handle Missing Values**: Remove or impute missing data before upload
3. **Feature Selection**: Start with fewer features for faster optimization
4. **Test Size**: Use 20-30% test size for better generalization

### Optimization

1. **Start Small**: Begin with 10-20 trials for testing
2. **Increase Gradually**: Use 100+ trials for final optimization
3. **Quantum Models**: Require more time, use fewer trials initially
4. **Classical Baseline**: Always compare with classical models

### Workflow Design

1. **Linear Flows**: Start simple with linear pipelines
2. **Validation**: Use Data Preview to verify data loading
3. **Save Work**: Export models and results regularly
4. **Documentation**: Use descriptive workflow names

## Troubleshooting

### Common Issues

#### Node Not Executing
- **Check connections**: Ensure all input edges are connected
- **Verify order**: Parent nodes must complete before children
- **Review config**: Some nodes require configuration

#### Upload Fails
- **File format**: Only CSV files are supported
- **File size**: Must be under 100 MB
- **Format**: Ensure proper CSV structure (comma-separated)

#### Optimization Slow
- **Reduce trials**: Lower n_trials for faster results
- **Use classical**: Quantum models are inherently slower
- **Check dataset**: Larger datasets take longer

#### SHAP Analysis Fails
- **Model compatibility**: Not all models support SHAP
- **Data size**: Very large datasets may timeout
- **Try subsampling**: Use fewer samples for SHAP

### Error Messages

- **"Workflow contains cycles"**: Remove circular dependencies
- **"No input data"**: Connect required input nodes
- **"No model name provided"**: Configure model selection
- **"Execution failed"**: Check backend logs for details

## API Integration

### REST API Endpoints

The workflow builder uses these backend endpoints:

```
POST   /api/v1/data/upload              # Upload CSV file
GET    /api/v1/data/uci                 # List UCI datasets
POST   /api/v1/workflows/execute        # Execute workflow
GET    /api/v1/workflows/executions/:id # Get execution status
```

### Execution Response Format

```json
{
  "execution_id": "exec-1",
  "status": "pending",
  "message": "Workflow execution started"
}
```

### Status Response Format

```json
{
  "id": "exec-1",
  "status": "completed",
  "started_at": "2025-11-15T00:00:00",
  "completed_at": "2025-11-15T00:05:00",
  "result": {
    "status": "completed",
    "workflow_name": "My Workflow",
    "node_results": { /* ... */ }
  }
}
```

## Advanced Features

### Custom Configurations

Nodes can be configured by modifying their `config` property in the workflow JSON:

```json
{
  "id": "node-1",
  "type": "optuna-config",
  "data": {
    "type": "optuna-config",
    "label": "Optuna Config",
    "config": {
      "study_name": "my_study",
      "n_trials": 200,
      "db_name": "custom.db"
    }
  }
}
```

### Programmatic Access

You can also execute workflows programmatically via the API:

```javascript
import { executeWorkflow, pollExecutionStatus } from './lib/api';

const result = await executeWorkflow({
  name: 'Automated Workflow',
  nodes: [...],
  edges: [...]
});

const status = await pollExecutionStatus(result.execution_id);
console.log(status.result);
```

## Future Enhancements

Planned features for future releases:

- [ ] Node configuration UI panels
- [ ] Workflow templates and examples
- [ ] Real-time progress streaming
- [ ] Workflow sharing and export
- [ ] Custom node plugins
- [ ] Parallel execution paths
- [ ] Model versioning and tracking
- [ ] Interactive result visualization
- [ ] Workflow scheduling
- [ ] Multi-objective optimization

## Support

For issues, questions, or feature requests:

- Check the backend logs: `docker compose logs backend`
- View frontend console for client errors
- API documentation: http://localhost:8000/docs
- Submit issues on GitHub

## License

QuOptuna is licensed under the Apache-2.0 License.
