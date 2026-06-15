# QuOptuna Workflow Builder Testing Checklist

## Test Environment Setup

### Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] Ports 5173 and 8000 available
- [ ] Sample CSV file prepared for upload testing

### Start Services
```bash
cd /home/user/quoptuna
docker compose up --build
```

- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Can access http://localhost:5173
- [ ] Can access http://localhost:8000/docs

## Backend API Tests

### Data Upload Endpoint

**Test 1: Upload Valid CSV**
```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@test_data.csv"
```
- [ ] Returns 201 status code
- [ ] Returns file_id, file_path, rows, and columns
- [ ] File is saved in ./uploads directory
- [ ] Can read uploaded file

**Test 2: Upload Invalid File**
```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@test_data.txt"
```
- [ ] Returns 400 error
- [ ] Error message: "Only CSV files are allowed"

**Test 3: Upload Malformed CSV**
```bash
# Create a CSV with invalid format
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@malformed.csv"
```
- [ ] Returns 400 error
- [ ] Error message indicates CSV parsing failure
- [ ] No file is left in uploads directory

### Workflow Execution Endpoint

**Test 4: Execute Simple Workflow**
```bash
curl -X POST http://localhost:8000/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Workflow",
    "nodes": [
      {
        "id": "node-1",
        "data": {
          "type": "data-uci",
          "label": "UCI Dataset",
          "config": {"dataset_id": "53"}
        }
      }
    ],
    "edges": []
  }'
```
- [ ] Returns execution_id
- [ ] Status is "pending"
- [ ] Can poll execution status

**Test 5: Get Execution Status**
```bash
curl http://localhost:8000/api/v1/workflows/executions/{execution_id}
```
- [ ] Returns execution details
- [ ] Status transitions: pending → running → completed
- [ ] Result contains workflow output

## Frontend UI Tests

### Navigation and Layout

- [ ] Sidebar menu is visible
- [ ] Can navigate to Dashboard
- [ ] Can navigate to Workflow Builder
- [ ] Can navigate to Data Explorer
- [ ] Can navigate to Models
- [ ] Can navigate to Analytics
- [ ] Can navigate to Settings
- [ ] Current page is highlighted in sidebar

### Node Palette

- [ ] Node Palette is visible on left side
- [ ] All 6 categories are displayed:
  - [ ] Data (4 nodes)
  - [ ] Preprocessing (3 nodes)
  - [ ] Models (2 nodes)
  - [ ] Optimization (2 nodes)
  - [ ] Analysis (3 nodes)
  - [ ] Output (2 nodes)
- [ ] Each node shows label and description
- [ ] Nodes are draggable
- [ ] Click on node also adds it to canvas

### Canvas Operations

**Test 6: Add Nodes**
- [ ] Drag "Upload CSV" node onto canvas
- [ ] Node appears at drop location
- [ ] Node has unique ID
- [ ] Node displays correct label
- [ ] Can drag multiple nodes
- [ ] Each node is independently selectable

**Test 7: Connect Nodes**
- [ ] Add "UCI Dataset" node
- [ ] Add "Data Preview" node
- [ ] Drag from UCI Dataset output to Data Preview input
- [ ] Edge is created and visible
- [ ] Edge has arrow showing direction
- [ ] Edge connects correct handles

**Test 8: Delete Elements**
- [ ] Select a node
- [ ] Press Delete key (or Backspace)
- [ ] Node is removed from canvas
- [ ] Connected edges are also removed
- [ ] Select an edge
- [ ] Press Delete key
- [ ] Edge is removed

**Test 9: Canvas Controls**
- [ ] Zoom in with mouse wheel
- [ ] Zoom out with mouse wheel
- [ ] Pan by dragging on empty canvas
- [ ] Fit view button centers all nodes
- [ ] Mini-map shows overview (if implemented)

### Toolbar Actions

**Test 10: Clear Canvas**
- [ ] Add several nodes to canvas
- [ ] Click "Clear" button
- [ ] Confirmation dialog appears
- [ ] All nodes and edges are removed

**Test 11: Save Workflow** (if implemented)
- [ ] Create a workflow
- [ ] Click "Save" button
- [ ] Workflow is saved
- [ ] Can reload saved workflow

**Test 12: Load Workflow** (if implemented)
- [ ] Click "Load" button
- [ ] Saved workflows are listed
- [ ] Select a workflow
- [ ] Canvas is populated with nodes and edges

## Node Functionality Tests

### Data Nodes

**Test 13: UCI Dataset Node**
1. Add UCI Dataset node
2. Configure with dataset_id: 53 (Iris)
3. Connect to Data Preview node
4. Run workflow
- [ ] Dataset is fetched successfully
- [ ] Data Preview shows 150 rows, 5 columns
- [ ] Execution completes without errors

**Test 14: Upload CSV Node**
1. Add Upload CSV node
2. Upload a test CSV file
3. Connect to Data Preview node
4. Run workflow
- [ ] File uploads successfully
- [ ] File path is stored in node config
- [ ] Data Preview shows correct dimensions
- [ ] Execution completes without errors

**Test 15: Data Preview Node**
1. Connect data source to Data Preview
2. Run workflow
- [ ] Preview shows shape (rows, columns)
- [ ] Shows data types for each column
- [ ] Shows statistical summary
- [ ] Shows first few rows

**Test 16: Feature Selection Node**
1. Connect dataset to Feature Selection
2. Configure x_columns and y_column
3. Connect to Train/Test Split
4. Run workflow
- [ ] Features are correctly separated
- [ ] X contains specified columns
- [ ] y contains target column
- [ ] Execution continues to next node

### Preprocessing Nodes

**Test 17: Train/Test Split Node**
1. Connect Feature Selection to Train/Test Split
2. Run workflow
- [ ] Data is split into train and test sets
- [ ] x_train, x_test, y_train, y_test are created
- [ ] Split ratio is approximately 80/20
- [ ] Data is automatically scaled

**Test 18: Scaler and Encoding Nodes**
1. Add Scaler and Label Encoding nodes
2. Connect in preprocessing pipeline
3. Run workflow
- [ ] Nodes execute successfully
- [ ] Data passes through (handled by DataPreparation)
- [ ] No errors occur

### Model and Optimization Nodes

**Test 19: Quantum Model Node**
1. Add Quantum Model node after preprocessing
2. Configure model_name: "DataReuploading"
3. Connect to Optuna Config
4. Run workflow
- [ ] Model configuration is stored
- [ ] model_type is "quantum"
- [ ] Execution continues

**Test 20: Classical Model Node**
1. Add Classical Model node after preprocessing
2. Configure model_name: "RandomForest"
3. Connect to Optuna Config
4. Run workflow
- [ ] Model configuration is stored
- [ ] model_type is "classical"
- [ ] Execution continues

**Test 21: Optuna Config Node**
1. Add Optuna Config node
2. Configure:
   - study_name: "test_study"
   - n_trials: 10
   - db_name: "test.db"
3. Connect to Run Optimization
4. Run workflow
- [ ] Configuration is merged with model config
- [ ] Parameters are passed to optimizer
- [ ] Execution continues

**Test 22: Run Optimization Node**
1. Complete workflow: Data → Features → Split → Model → Optuna → Optimization
2. Configure for 10 trials (for speed)
3. Run workflow
- [ ] Optimization starts
- [ ] Status updates show "running"
- [ ] Trials are executed (check backend logs)
- [ ] Best parameters are found
- [ ] best_value, best_params, best_trial_number returned
- [ ] SQLite database is created
- [ ] Execution completes successfully

### Analysis Nodes

**Test 23: SHAP Analysis Node**
1. Connect Optimization results to SHAP Analysis
2. Configure plot_types: ["bar", "beeswarm"]
3. Run workflow
- [ ] SHAP analysis executes
- [ ] Plots are generated (check result)
- [ ] Feature importance is calculated
- [ ] No errors occur

**Test 24: Feature Importance Node**
1. Connect SHAP results to Feature Importance
2. Run workflow
- [ ] Feature rankings are displayed
- [ ] Uses SHAP data if available
- [ ] Execution completes

## Complete Workflow Tests

### Test 25: End-to-End Classification Workflow

**Workflow Structure:**
```
UCI Dataset (Iris)
  ↓
Select Features (all but target → species)
  ↓
Train/Test Split
  ↓
Quantum Model (DataReuploading)
  ↓
Optuna Config (20 trials)
  ↓
Run Optimization
  ↓
SHAP Analysis
```

**Execution:**
1. Build workflow by connecting nodes
2. Click Run
3. Monitor execution progress

**Verification:**
- [ ] All nodes execute in correct order
- [ ] Status updates show progress
- [ ] Dataset is loaded (150 rows)
- [ ] Features are selected and split
- [ ] Optimization runs for 20 trials
- [ ] SHAP analysis completes
- [ ] Final result contains all node outputs
- [ ] Execution time is reasonable (< 10 minutes for 20 trials)
- [ ] All nodes show green (completed) status

### Test 26: Classical Model Comparison

**Create two parallel paths:**
1. Path 1: Quantum Model (DataReuploading)
2. Path 2: Classical Model (RandomForest)
3. Both paths from same data source
4. Both with 20 trials optimization

**Execution:**
- [ ] Workflow executes both paths
- [ ] Results can be compared
- [ ] No conflicts between parallel executions
- [ ] Both optimizations complete successfully

### Test 27: Upload Custom Dataset

**Workflow:**
```
Upload CSV
  ↓
Data Preview
  ↓
Select Features
  ↓
Train/Test Split
  ↓
Classical Model (SVC)
  ↓
Optuna Config (10 trials)
  ↓
Run Optimization
```

**Steps:**
1. Prepare a custom classification dataset CSV
2. Upload via Upload CSV node
3. Preview to verify columns
4. Select appropriate features and target
5. Run complete workflow

**Verification:**
- [ ] File uploads successfully
- [ ] Preview shows correct data
- [ ] Features are selected properly
- [ ] Optimization runs with custom data
- [ ] Results are meaningful for the dataset

## Error Handling Tests

**Test 28: Circular Dependency**
1. Create nodes: A → B → C → A
2. Run workflow
- [ ] Error: "Workflow contains cycles"
- [ ] Execution stops
- [ ] Clear error message displayed

**Test 29: Missing Input**
1. Add Optimization node without connecting data
2. Run workflow
- [ ] Error: "No input configuration for optimization"
- [ ] Node status shows failed
- [ ] Error message is clear

**Test 30: Invalid Configuration**
1. Feature Selection without specifying columns
2. Run workflow
- [ ] Error: "Must specify x_columns and y_column"
- [ ] Execution fails at that node
- [ ] Subsequent nodes don't execute

## Performance Tests

**Test 31: Large Dataset**
1. Upload CSV with 10,000+ rows
2. Run optimization with 50 trials
- [ ] Execution doesn't timeout
- [ ] Progress updates continue
- [ ] Memory usage is acceptable
- [ ] Execution completes (may take a while)

**Test 32: Multiple Simultaneous Workflows**
1. Start workflow execution
2. Immediately start another workflow
- [ ] Both execute in background
- [ ] No conflicts between executions
- [ ] Both complete successfully
- [ ] Execution IDs are unique

**Test 33: Rapid Node Addition**
1. Quickly drag 20+ nodes onto canvas
2. Connect them in sequence
3. Run workflow
- [ ] UI remains responsive
- [ ] All nodes are tracked
- [ ] Execution proceeds correctly

## Integration Tests

**Test 34: Backend Restart During Execution**
1. Start a long-running optimization
2. Restart backend container
3. Check execution status
- [ ] Execution state is handled gracefully
- [ ] Error message if execution is lost
- [ ] New executions work after restart

**Test 35: Frontend Refresh During Execution**
1. Start workflow execution
2. Refresh browser page
3. Check execution status
- [ ] Execution continues in backend
- [ ] Can query status by execution_id (if stored)
- [ ] UI state can be recovered

## Browser Compatibility

Test in different browsers:
- [ ] Chrome/Chromium - All features work
- [ ] Firefox - All features work
- [ ] Safari - All features work
- [ ] Edge - All features work

## Accessibility Tests

- [ ] Keyboard navigation works
- [ ] Tab order is logical
- [ ] Can operate without mouse
- [ ] Screen reader compatibility (basic)
- [ ] Sufficient color contrast
- [ ] Focus indicators are visible

## Documentation Tests

**Test 36: User Guide**
- [ ] Guide is accurate for all features
- [ ] Examples can be reproduced
- [ ] API documentation matches implementation
- [ ] Troubleshooting section is helpful

**Test 37: API Documentation**
- [ ] Access http://localhost:8000/docs
- [ ] All endpoints are documented
- [ ] Can test endpoints from Swagger UI
- [ ] Request/response schemas are correct

## Known Issues / Limitations

Document any issues found during testing:

1. **Node Configuration UI**: Not yet implemented - config must be set programmatically
2. **Export Model**: Returns placeholder - actual export not implemented
3. **Generate Report**: Requires LLM API keys - returns placeholder
4. **Confusion Matrix**: Not fully implemented - returns placeholder
5. **Workflow Persistence**: No database - workflows lost on reload
6. **Parallel Execution**: Workflows run in sequence, not true parallel processing

## Test Data Requirements

### Sample CSVs Needed

**1. iris.csv** - Small classification dataset
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
...
```

**2. binary_classification.csv** - Binary target
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

**3. multi_feature.csv** - Many features (10+)
```csv
f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,target
...
```

## Test Results Summary

| Test # | Feature | Status | Notes |
|--------|---------|--------|-------|
| 1 | Upload Valid CSV | ⏳ | Pending test |
| 2 | Upload Invalid File | ⏳ | Pending test |
| ... | ... | ... | ... |

**Legend:**
- ✅ Passed
- ❌ Failed
- ⏳ Pending
- ⚠️ Partial

## Sign-off

- **Tested By**: _______________
- **Date**: _______________
- **Environment**: Docker Compose / Local
- **Overall Status**: ⏳ Pending / ✅ Passed / ❌ Failed
- **Ready for Production**: Yes / No

## Next Steps

After completing all tests:

1. Fix any failing tests
2. Document known limitations
3. Create user acceptance tests
4. Plan production deployment
5. Set up monitoring and logging
6. Create backup and recovery procedures
