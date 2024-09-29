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
from torch.utils.data import DataLoader 
from src.circuits.convolution import custom_circuit
from src.circuits.embedding import custom_embedding
from src.circuits.measurement import custom_measurement

import numpy as np 

# Assuming 'qcml' is installed and accessible
try:
    from qcml.bench.grid_search import GridSearch
    from qcml.utils.log import log_setup
except ImportError:
    GridSearch = None

from src.models import QuanvolutionalNet
from src.utils import load_mnist_data

def test_grid_search_import():
    assert GridSearch is not None

def test_load_mnist_data_dl():
    train_loader, test_loader = load_mnist_data(batch_size=4, output='dl', limit=50)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader.dataset) == 50
    assert len(test_loader.dataset) == 50

def test_load_mnist_data_np():
    X_train, y_train, X_test, y_test = load_mnist_data(batch_size=4, output='np', limit=50)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert len(X_train) == 50
    assert len(y_train) == 50
    assert len(X_test) == 50
    assert len(y_test) == 50

def test_initialize_model():
    qkernel_shape = 2
    classical_kernel_shape = 3
    n_classes = 10
    num_layers = 2
    num_wires = qkernel_shape ** 2
    weights = torch.randn(num_layers, num_wires, 3)

    params = {'circuit': {'weights': weights}}

    model = QuanvolutionalNet(
        qkernel_shape=qkernel_shape,
        classical_kernel_shape=classical_kernel_shape,
        embedding=custom_embedding,
        circuit=custom_circuit,
        measurement=custom_measurement,
        params=params,
        n_classes=n_classes
    )

    assert model is not None

def test_qcml_grid_search():
    if GridSearch is None:
        pytest.skip("qcml is not installed")

    # Setup logging if necessary
    log_settings = {
        "output": "both",
        "terminal_level": "INFO",
        "file_level": "DEBUG",
        "hide_logs": ["jax", "pennylane", "bokeh", "distributed"],
        "slack_notify": False,
        "slack_credentials": ["YOUR_SLACK_TOKEN", "#qcml"],
    }

    log_setup(**log_settings)

    X_train, y_train, X_val, y_val = load_mnist_data(batch_size=4, output='np', limit=50)

    combinations = [
        [{'qkernel_shape': 2, 'classical_kernel_shape': 3, 'n_classes': 10}, None, None]
    ]

    gs = GridSearch(
        classifiers=[QuanvolutionalNet],
        combinations=combinations,
        batch_size=10,
        experiment_name="test_qcml_grid_search"
    )

    try:
        gs.run(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    except Exception as e:
        pytest.fail(f"QCML GridSearch raised an exception: {e}")
