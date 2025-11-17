"""
Workflow execution service

This service handles the execution of visual workflows created in the frontend.
It integrates with the existing quoptuna services (Optimizer, DataPreparation, XAI).
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

from quoptuna import Optimizer, DataPreparation, XAI, XAIConfig, create_model

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Raised when workflow execution fails"""
    pass


class WorkflowExecutor:
    """Executes visual workflows by running nodes in topological order"""

    def __init__(self, workflow: Dict[str, Any], upload_dir: str = "./uploads"):
        self.workflow = workflow
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        self.nodes = {node["id"]: node for node in workflow.get("nodes", [])}
        self.edges = workflow.get("edges", [])
        self.results = {}  # Store results from each node

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get list of node IDs that this node depends on"""
        dependencies = []
        for edge in self.edges:
            if edge["target"] == node_id:
                dependencies.append(edge["source"])
        return dependencies

    def topological_sort(self) -> List[str]:
        """Sort nodes in execution order using topological sort"""
        # Build dependency graph
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge["target"]] += 1

        # Find nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        sorted_nodes = []

        while queue:
            node_id = queue.pop(0)
            sorted_nodes.append(node_id)

            # Reduce in-degree for dependent nodes
            for edge in self.edges:
                if edge["source"] == node_id:
                    in_degree[edge["target"]] -= 1
                    if in_degree[edge["target"]] == 0:
                        queue.append(edge["target"])

        if len(sorted_nodes) != len(self.nodes):
            raise WorkflowExecutionError("Workflow contains cycles")

        return sorted_nodes

    def execute_node(self, node_id: str) -> Any:
        """Execute a single node and return its result"""
        node = self.nodes[node_id]
        node_type = node["data"]["type"]
        config = node["data"].get("config", {})

        logger.info(f"Executing node {node_id} of type {node_type}")

        # Get input from dependencies
        dependencies = self.get_node_dependencies(node_id)
        inputs = {dep: self.results[dep] for dep in dependencies}

        # Execute based on node type
        if node_type == "data-upload":
            return self._execute_data_upload(config, inputs)
        elif node_type == "data-uci":
            return self._execute_data_uci(config, inputs)
        elif node_type == "data-preview":
            return self._execute_data_preview(config, inputs)
        elif node_type == "feature-selection":
            return self._execute_feature_selection(config, inputs)
        elif node_type == "train-test-split":
            return self._execute_train_test_split(config, inputs)
        elif node_type == "scaler":
            return self._execute_scaler(config, inputs)
        elif node_type == "label-encoding":
            return self._execute_label_encoding(config, inputs)
        elif node_type in ["quantum-model", "classical-model"]:
            return self._execute_model_config(config, inputs, node_type)
        elif node_type == "optuna-config":
            return self._execute_optuna_config(config, inputs)
        elif node_type == "optimization":
            return self._execute_optimization(config, inputs)
        elif node_type == "shap-analysis":
            return self._execute_shap_analysis(config, inputs)
        elif node_type == "confusion-matrix":
            return self._execute_confusion_matrix(config, inputs)
        elif node_type == "feature-importance":
            return self._execute_feature_importance(config, inputs)
        elif node_type == "export-model":
            return self._execute_export_model(config, inputs)
        elif node_type == "generate-report":
            return self._execute_generate_report(config, inputs)
        else:
            raise WorkflowExecutionError(f"Unknown node type: {node_type}")

    def _execute_data_upload(self, config: Dict, inputs: Dict) -> Dict:
        """Handle CSV file upload"""
        file_path = config.get("file_path")
        if not file_path:
            raise WorkflowExecutionError("No file path provided for data upload")

        df = pd.read_csv(file_path)
        return {
            "type": "dataset",
            "dataframe": df,
            "rows": len(df),
            "columns": list(df.columns),
        }

    def _execute_data_uci(self, config: Dict, inputs: Dict) -> Dict:
        """Fetch dataset from UCI repository"""
        dataset_id = config.get("dataset_id")
        if not dataset_id:
            raise WorkflowExecutionError("No dataset ID provided")

        # Fetch from UCI
        dataset = fetch_ucirepo(id=int(dataset_id))
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

        return {
            "type": "dataset",
            "dataframe": df,
            "rows": len(df),
            "columns": list(df.columns),
            "metadata": dataset.metadata,
        }

    def _execute_data_preview(self, config: Dict, inputs: Dict) -> Dict:
        """Generate dataset preview statistics"""
        if not inputs:
            raise WorkflowExecutionError("No input data for preview")

        dataset = list(inputs.values())[0]
        df = dataset["dataframe"]

        # Convert dtypes to strings for JSON serialization
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Convert describe() output, handling NaN values
        describe_dict = df.describe().fillna(0).to_dict()

        return {
            "type": "preview",
            "shape": list(df.shape),  # Convert tuple to list
            "dtypes": dtypes_dict,
            "describe": describe_dict,
            "head": df.head().to_dict(),
            "columns": list(df.columns),
            "rows": len(df),
        }

    def _execute_feature_selection(self, config: Dict, inputs: Dict) -> Dict:
        """Select features and target column"""
        if not inputs:
            raise WorkflowExecutionError("No input data for feature selection")

        dataset = list(inputs.values())[0]
        df = dataset["dataframe"]

        x_columns = config.get("x_columns", [])
        y_column = config.get("y_column")

        if not x_columns or not y_column:
            raise WorkflowExecutionError("Must specify x_columns and y_column")

        return {
            "type": "selected_data",
            "x": df[x_columns],
            "y": df[y_column],
            "x_columns": x_columns,
            "y_column": y_column,
        }

    def _execute_train_test_split(self, config: Dict, inputs: Dict) -> Dict:
        """Split data into train and test sets"""
        if not inputs:
            raise WorkflowExecutionError("No input data for train/test split")

        data = list(inputs.values())[0]
        x = data["x"]
        y = data["y"]

        # Use DataPreparation class
        data_prep = DataPreparation(
            dataset={"x": x, "y": y},
            x_cols=list(x.columns),
            y_col=data["y_column"] if isinstance(data["y_column"], str) else data["y_column"][0]
        )

        return {
            "type": "split_data",
            "x_train": data_prep.x_train,
            "x_test": data_prep.x_test,
            "y_train": data_prep.y_train,
            "y_test": data_prep.y_test,
            "x_columns": data["x_columns"],
            "y_column": data["y_column"],
        }

    def _execute_scaler(self, config: Dict, inputs: Dict) -> Dict:
        """Data is already scaled by DataPreparation, just pass through"""
        if not inputs:
            raise WorkflowExecutionError("No input data for scaler")

        # DataPreparation already handles scaling, just pass through
        return list(inputs.values())[0]

    def _execute_label_encoding(self, config: Dict, inputs: Dict) -> Dict:
        """Encode labels to -1 and 1 for binary classification (as required by quantum models)"""
        if not inputs:
            raise WorkflowExecutionError("No input data for label encoding")

        result = list(inputs.values())[0]

        # Get unique classes from training data
        import numpy as np
        y_train = result["y_train"]
        y_test = result["y_test"]

        # Get unique classes
        unique_classes = np.unique(np.concatenate([
            y_train.values.ravel() if hasattr(y_train, 'values') else y_train.ravel(),
            y_test.values.ravel() if hasattr(y_test, 'values') else y_test.ravel()
        ]))

        # For binary classification, map to -1 and 1
        if len(unique_classes) == 2:
            logger.info(f"Binary classification detected. Mapping classes {unique_classes} to [-1, 1]")

            # Create mapping: first class -> -1, second class -> 1
            class_mapping = {unique_classes[0]: -1, unique_classes[1]: 1}

            # Apply mapping
            if hasattr(y_train, 'replace'):
                # pandas DataFrame/Series
                result["y_train"] = y_train.replace(class_mapping)
                result["y_test"] = y_test.replace(class_mapping)
            else:
                # numpy array
                y_train_mapped = np.where(y_train == unique_classes[0], -1, 1)
                y_test_mapped = np.where(y_test == unique_classes[0], -1, 1)
                result["y_train"] = pd.DataFrame(y_train_mapped, columns=y_train.columns if hasattr(y_train, 'columns') else ['target'])
                result["y_test"] = pd.DataFrame(y_test_mapped, columns=y_test.columns if hasattr(y_test, 'columns') else ['target'])
        else:
            logger.warning(f"Multi-class classification detected ({len(unique_classes)} classes). Models may not support this.")

        return result

    def _execute_model_config(self, config: Dict, inputs: Dict, node_type: str) -> Dict:
        """Configure model selection"""
        model_name = config.get("model_name")
        if not model_name:
            raise WorkflowExecutionError("No model name provided")

        result = {
            "type": "model_config",
            "model_name": model_name,
            "model_type": "quantum" if node_type == "quantum-model" else "classical",
        }

        # Merge input data if available
        if inputs:
            result.update(list(inputs.values())[0])

        return result

    def _execute_optuna_config(self, config: Dict, inputs: Dict) -> Dict:
        """Configure Optuna optimization parameters"""
        result = {
            "type": "optuna_config",
            "study_name": config.get("study_name", "workflow_study"),
            "n_trials": config.get("n_trials", 100),
            "db_name": config.get("db_name", "workflow_optimization.db"),
        }

        # Merge input data if available
        if inputs:
            result.update(list(inputs.values())[0])

        return result

    def _execute_optimization(self, config: Dict, inputs: Dict) -> Dict:
        """Run Optuna optimization"""
        if not inputs:
            raise WorkflowExecutionError("No input configuration for optimization")

        opt_config = list(inputs.values())[0]

        # Store original DataFrames for later SHAP analysis
        x_train_df = opt_config["x_train"]
        x_test_df = opt_config["x_test"]
        y_train_df = opt_config["y_train"]
        y_test_df = opt_config["y_test"]

        # Convert to numpy arrays for Optimizer (as shown in notebooks)
        data_dict = {
            "train_x": x_train_df.values if hasattr(x_train_df, 'values') else x_train_df,
            "train_y": y_train_df.values if hasattr(y_train_df, 'values') else y_train_df,
            "test_x": x_test_df.values if hasattr(x_test_df, 'values') else x_test_df,
            "test_y": y_test_df.values if hasattr(y_test_df, 'values') else y_test_df,
        }

        # Create optimizer
        optimizer = Optimizer(
            db_name=opt_config.get("db_name", "workflow_optimization.db"),
            data=data_dict,
            study_name=opt_config.get("study_name", "workflow_study"),
        )

        # Run optimization
        n_trials = opt_config.get("n_trials", 100)
        model_name = opt_config.get("model_name", "DataReuploading")

        # Note: model_name is stored for reference, but Optuna will try different models automatically
        optimizer.optimize(n_trials=n_trials)

        # Get best trial
        best_trial = optimizer.study.best_trial

        return {
            "type": "optimization_result",
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_trial_number": best_trial.number,
            "study_name": opt_config.get("study_name"),
            "db_name": opt_config.get("db_name"),
            "n_trials": n_trials,
            "model_name": model_name,
            # Store DataFrames for SHAP analysis
            "x_train": x_train_df,
            "x_test": x_test_df,
            "y_train": y_train_df,
            "y_test": y_test_df,
            "x_columns": opt_config.get("x_columns"),
            "y_column": opt_config.get("y_column"),
        }

    def _execute_shap_analysis(self, config: Dict, inputs: Dict) -> Dict:
        """Generate SHAP analysis"""
        if not inputs:
            raise WorkflowExecutionError("No input data for SHAP analysis")

        opt_result = list(inputs.values())[0]

        # Load the best model from Optuna study
        from optuna import load_study
        from quoptuna.backend.models import create_model
        import numpy as np

        db_name = opt_result.get("db_name")
        study_name = opt_result.get("study_name")

        storage_location = f"sqlite:///db/{db_name}.db"
        study = load_study(storage=storage_location, study_name=study_name)
        best_trial = study.best_trial

        # Get DataFrames from opt_result
        x_train_df = opt_result["x_train"]
        x_test_df = opt_result["x_test"]
        y_train_df = opt_result["y_train"]
        y_test_df = opt_result["y_test"]

        # Recreate and fit the best model (model.fit needs numpy arrays)
        model = create_model(best_trial.params["model_type"], **best_trial.params)

        # Convert to numpy for model fitting (as shown in notebooks)
        x_train_np = x_train_df.values if hasattr(x_train_df, 'values') else x_train_df
        y_train_np = y_train_df.values if hasattr(y_train_df, 'values') else y_train_df

        model.fit(x_train_np, y_train_np)

        # Prepare data dictionary for XAI (XAI expects DataFrames)
        data_dict = {
            "x_train": x_train_df,
            "x_test": x_test_df,
            "y_train": y_train_df,
            "y_test": y_test_df,
        }

        # Create XAI instance with XAIConfig
        xai_config = XAIConfig(use_proba=True, onsubset=True, subset_size=50)
        xai = XAI(model=model, data=data_dict, config=xai_config)

        # Generate SHAP plots
        plot_types = config.get("plot_types", ["bar", "beeswarm", "waterfall"])
        plots = {}

        for plot_type in plot_types:
            try:
                if plot_type == "bar":
                    plots[plot_type] = xai.get_bar_plot()
                elif plot_type == "beeswarm":
                    plots[plot_type] = xai.get_beeswarm_plot()
                elif plot_type == "waterfall":
                    plots[plot_type] = xai.get_waterfall_plot(index=0)
                elif plot_type == "violin":
                    plots[plot_type] = xai.get_violin_plot()
                elif plot_type == "heatmap":
                    plots[plot_type] = xai.get_heatmap_plot()
            except Exception as e:
                logger.error(f"Error generating {plot_type} plot: {e}")
                logger.exception("Full traceback:")

        # Get feature importance from SHAP values
        shap_values = xai.shap_values
        feature_importance = []

        if hasattr(shap_values, 'values') and hasattr(shap_values, 'data'):
            # Calculate mean absolute SHAP values for feature importance
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            # Get feature names from DataFrame columns
            feature_names = list(x_train_df.columns) if hasattr(x_train_df, 'columns') else [f"feature_{i}" for i in range(x_train_df.shape[1])]

            for i, feature in enumerate(feature_names):
                feature_importance.append({
                    "feature": feature,
                    "importance": float(mean_abs_shap[i]) if len(mean_abs_shap.shape) == 1 else float(mean_abs_shap[i].mean())
                })

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "type": "shap_analysis",
            "plots": plots,
            "feature_importance": feature_importance,
            "feature_names": feature_names if feature_importance else [],
        }

    def _execute_confusion_matrix(self, config: Dict, inputs: Dict) -> Dict:
        """Generate confusion matrix"""
        if not inputs:
            raise WorkflowExecutionError("No input data for confusion matrix")

        result = list(inputs.values())[0]
        return {
            "type": "confusion_matrix",
            "message": "Confusion matrix generation not yet implemented",
            **result,
        }

    def _execute_feature_importance(self, config: Dict, inputs: Dict) -> Dict:
        """Calculate feature importance"""
        if not inputs:
            raise WorkflowExecutionError("No input data for feature importance")

        result = list(inputs.values())[0]

        if "plots" in result and "bar" in result["plots"]:
            # SHAP bar plot already shows feature importance
            return {
                "type": "feature_importance",
                "source": "shap",
                **result,
            }

        return {
            "type": "feature_importance",
            "message": "Feature importance analysis",
            **result,
        }

    def _execute_export_model(self, config: Dict, inputs: Dict) -> Dict:
        """Export trained model"""
        if not inputs:
            raise WorkflowExecutionError("No input data for model export")

        result = list(inputs.values())[0]
        export_path = config.get("export_path", f"./models/{result.get('study_name', 'model')}.pkl")

        return {
            "type": "model_export",
            "export_path": export_path,
            "message": f"Model would be exported to {export_path}",
            **result,
        }

    def _execute_generate_report(self, config: Dict, inputs: Dict) -> Dict:
        """Generate AI-powered report"""
        if not inputs:
            raise WorkflowExecutionError("No input data for report generation")

        result = list(inputs.values())[0]
        llm_provider = config.get("llm_provider", "openai")

        return {
            "type": "report",
            "llm_provider": llm_provider,
            "message": "AI report generation requires LLM API keys to be configured",
            **result,
        }

    def execute(self) -> Dict[str, Any]:
        """Execute the entire workflow"""
        logger.info(f"Starting workflow execution: {self.workflow.get('name', 'Unnamed')}")

        try:
            # Get execution order
            execution_order = self.topological_sort()

            # Execute nodes in order
            for node_id in execution_order:
                result = self.execute_node(node_id)
                self.results[node_id] = result
                logger.info(f"Node {node_id} executed successfully")

            # Return final results
            final_result = {
                "status": "completed",
                "workflow_id": self.workflow.get("id"),
                "workflow_name": self.workflow.get("name"),
                "execution_order": execution_order,
                "node_results": self.results,
            }

            logger.info("Workflow execution completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowExecutionError(f"Execution failed: {str(e)}") from e
