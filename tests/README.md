# Unit Tests

This directory contains unit tests for the SwinCup project.

## Running Tests

The tests are written using Python's built-in `unittest` module. To discover and run all tests, navigate to the root directory of the project and execute:

```bash
python -m unittest discover tests
```

Alternatively, you can run individual test files:

```bash
python -m unittest tests.test_dataio
python -m unittest tests.test_models
# etc.
```

## Test Structure

- `test_dataio.py`: Contains tests for data loading, dataset classes, and transformations.
- `test_models.py`: Contains tests for model instantiation and forward passes.
- `test_loss.py`: (If applicable) Contains tests for custom loss functions.

## Coverage

The tests aim to cover:
- Data loading and preprocessing logic.
- Forward pass execution for key models.
- Behavior of custom loss functions.

These tests are designed to run on CPU and, where specified, also on GPU if available.
