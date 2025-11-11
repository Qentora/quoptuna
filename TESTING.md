# Quoptuna Testing Guide

## Overview

This document describes the test suite for Quoptuna, a quantum computing and hyperparameter optimization framework. The test suite emphasizes **fixture-based testing** over mocks to ensure solid integration testing with real data.

## Test Structure

### Test Files

1. **conftest.py** - Shared pytest fixtures
   - Classification dataset fixtures
   - Train/test split fixtures
   - Trained model fixtures
   - Temporary file fixtures

2. **test_create_model.py** (22 tests)
   - Model creation tests for all quantum and classical models
   - Parameter validation tests
   - Error handling tests

3. **test_data_preparation.py** (2 tests)
   - Basic DataPreparation functionality
   - Column name handling

4. **test_data_preparation_extended.py** (20 tests)
   - Comprehensive DataPreparation class tests
   - Dataset initialization from files and objects
   - Data preprocessing and transformations
   - Custom scaler support
   - Output format variations

5. **test_data_utils.py** (10 tests)
   - Data loading and preprocessing utilities
   - CSV file handling
   - Port finding utilities
   - Mock data generation

6. **test_optimizer.py** (8 tests)
   - Optimizer initialization
   - Optimization runs with Optuna
   - Study loading and persistence
   - User attribute logging
   - Error handling

7. **test_xai.py** (13 tests)
   - XAI module initialization
   - SHAP explainer functionality
   - Plotting functions
   - Model validation

8. **test_xai_extended.py** (23 tests)
   - XAI configuration
   - Metrics computation (F1, precision, recall, MCC, etc.)
   - State serialization and deserialization
   - Custom data keys support

## Test Coverage

**Current Coverage: 38.43%**

### Module Coverage

| Module | Coverage | Lines Covered |
|--------|----------|---------------|
| `backend/models.py` | 100% | All lines |
| `backend/utils/data_utils/prepare.py` | 94% | 66/70 lines |
| `backend/utils/data_utils/data.py` | 81% | 34/42 lines |
| `backend/xai/xai.py` | 74% | 219/297 lines |
| `conftest.py` | 96% | 53/55 lines |

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_data_utils.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src/quoptuna --cov-report=html
```

### Run Without Slow Tests
```bash
pytest tests/ --ignore=tests/test_optimizer.py
```

## Design Principles

### 1. Fixture-Based Testing
All tests use pytest fixtures instead of mocks. This ensures:
- Real integration testing
- Realistic data flows
- Better bug detection
- More maintainable tests

### 2. Real Data
Tests use actual scikit-learn datasets or generated synthetic data:
```python
@pytest.fixture
def sample_classification_dataset():
    X, y = make_classification(n_samples=100, n_features=4, ...)
    y = np.where(y == 0, 1, -1)  # Binary labels
    return X, y
```

### 3. Temporary Files
Tests that need file I/O use pytest's `tmp_path` fixture:
```python
def test_csv_loading(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    # Test loading
```

### 4. Isolation
Each test is independent and can run in any order. Tests clean up after themselves using fixtures and context managers.

## Test Categories

### Unit Tests
- Individual function testing
- Class method testing
- Edge case handling

### Integration Tests
- End-to-end workflows
- Multi-component interactions
- Data pipeline testing

### Regression Tests
- Model creation validation
- Output format consistency
- API compatibility

## Adding New Tests

### 1. Create a Fixture (if needed)
```python
@pytest.fixture
def my_fixture():
    # Setup
    data = create_test_data()
    yield data
    # Teardown (if needed)
```

### 2. Write Test Function
```python
def test_my_feature(my_fixture):
    """Test description."""
    result = my_function(my_fixture)
    assert result is not None
    assert result.shape == (10, 5)
```

### 3. Follow Naming Conventions
- Test files: `test_*.py`
- Test functions: `test_*`
- Fixtures: descriptive names without `test_` prefix

## Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **One assertion focus** per test when possible
3. **Arrange-Act-Assert** pattern:
   ```python
   def test_example(fixture):
       # Arrange
       input_data = prepare_input(fixture)
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result == expected_value
   ```

4. **Test both success and failure** paths
5. **Use parametrize** for similar tests with different inputs:
   ```python
   @pytest.mark.parametrize("input,expected", [
       (1, 2),
       (2, 4),
       (3, 6),
   ])
   def test_doubling(input, expected):
       assert double(input) == expected
   ```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Main branch commits
- Scheduled nightly builds

Minimum coverage requirement: **35%**

## Test Dependencies

All test dependencies are included in the main project dependencies:
- pytest >= 8.3.4
- pytest-cov >= 6.0.0
- scikit-learn
- numpy
- pandas
- shap

## Known Issues

1. **Long-running optimizer tests**: Tests in `test_optimizer.py` can take 3+ minutes due to quantum model training
2. **SHAP plot tests**: Some SHAP plotting tests are sensitive to data shapes and dimensions
3. **Deprecation warnings**: PennyLane issues deprecation warnings for legacy devices

## Future Improvements

- [ ] Add property-based tests using Hypothesis
- [ ] Increase coverage to 50%+
- [ ] Add performance benchmarking tests
- [ ] Add tests for frontend components
- [ ] Add integration tests with real quantum backends
- [ ] Add load testing for optimization workflows

## Contact

For questions about testing, please:
- Open an issue on GitHub
- Check the project documentation
- Review existing test examples
