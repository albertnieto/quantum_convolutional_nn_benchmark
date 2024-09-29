# Copyright 2024 CTIC (Technological Center for Information and Communication).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import numpy as np
from src.scripts.train import load_data, initialize_model, train_model, evaluate_model, main
from torch.utils.data import DataLoader
from unittest.mock import patch
import io
import subprocess
import sys
import os

@pytest.fixture
def mock_data_np():
    # Load mock data in NumPy array format for testing purposes
    return load_data(output='np', limit=50)

@pytest.fixture
def mock_data_dl():
    # Load mock data in DataLoader format for testing purposes
    return load_data(output='dl', batch_size=8, limit=50)

def test_load_data_invalid_output():
    # Ensure that providing an unsupported output type raises a ValueError
    with pytest.raises(ValueError, match="Unsupported format: xyz. Use 'dl' or 'np'."):
        load_data(output='xyz', limit=50)

@pytest.fixture
def mock_model():
    # Initialize a model for testing purposes
    qkernel_shape = 2
    classical_kernel_shape = 3
    n_classes = 10
    num_layers = 2

    model = initialize_model(qkernel_shape, classical_kernel_shape, n_classes, num_layers)
    return model

def test_load_data_np(mock_data_np):
    X_train, y_train, X_test, y_test = mock_data_np
    # Check if the data is in NumPy format
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    # Check the sizes of the datasets
    assert X_train.shape[0] == 50
    assert y_train.shape[0] == 50
    assert X_test.shape[0] == 50
    assert y_test.shape[0] == 50

def test_load_data_dl(mock_data_dl):
    train_loader, test_loader = mock_data_dl
    # Check if the data is in DataLoader format
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    # Check that the datasets have the expected number of items
    assert len(train_loader.dataset) == 50
    assert len(test_loader.dataset) == 50

def test_initialize_model():
    qkernel_shape = 2
    classical_kernel_shape = 3
    n_classes = 10
    num_layers = 2

    model = initialize_model(qkernel_shape, classical_kernel_shape, n_classes, num_layers)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_train_model(mock_model, mock_data_np):
    X_train, y_train, _, _ = mock_data_np
    model = mock_model

    # Run training for testing purposes
    train_model(model, X_train, y_train, epochs=1, batch_size=8)

    assert model is not None  # Training didn't crash

def test_evaluate_model(mock_model, mock_data_np):
    _, _, X_test, y_test = mock_data_np
    model = mock_model

    # Perform mock training before evaluation
    train_model(model, X_test, y_test, epochs=1, batch_size=8)

    # Run evaluation
    accuracy = evaluate_model(model, X_test, y_test)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0  # Valid accuracy range

@patch('sys.stdout', new_callable=io.StringIO)
def test_main(mock_stdout):
    # Run main function with controlled parameters for testing
    accuracy = main(batch_size=4, limit=50, epochs=1, batch_size_train=8, suppress_print=False)

    # Check if the accuracy returned by main is valid
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

    # Verify that the print statement was called
    output = mock_stdout.getvalue().strip()
    assert "Test Accuracy:" in output

def test_script_execution():
    # Get the script path
    script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts', 'train.py')
    # Set the PYTHONPATH environment variable to include the root of the project
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Run the script directly using subprocess to simulate the "__main__" behavior
    result = subprocess.run(
        [sys.executable, script_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        env=env
    )

    # Check that the script executed successfully
    assert result.returncode == 0

    # Verify that the output contains the accuracy statement
    assert "Test Accuracy:" in result.stdout